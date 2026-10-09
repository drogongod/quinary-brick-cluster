#!/usr/bin/env python3
r"""lutgtlm_ingest.py -- CONTINUOUS conversation ingestion into the LutGTLM local learner.
The opposite of a frozen base + LoRA adapter: a PERSISTENT native base-5 mind whose WEIGHTS load from disk, learn
whatever text you feed it (local BPTT-through-time, no framework, no frozen base, no adapter), and save back. Run it
again with more text and it KEEPS LEARNING into the same weights. This is the "forever-learning, uncrystallized"
path -- adding conversations changes WHO SHE IS (the weights), not a bolt-on.

Usage:
  python lutgtlm_ingest.py <source.jsonl|source.txt> [--passes N]     # ingest a file into the persistent mind
  python lutgtlm_ingest.py --say "i am"                                # sample what it has learned so far
Persistent mind at:  C:/galatea_gtlm/lutgtlm/mind.npz   (delete it to start a fresh mind)

ARCHITECTURE, 2026-10-09 -- BRICK CLUSTER, replacing the single-D=384-net native_quinary_v1 path:
many small (D=40, ~10k param), FULLY trained quinary bricks instead of one large net, routed by a
real LSL-style (arXiv:2610.02126) density gate -- not an oracle, not a fixed K. A line's windows
train whichever existing brick's own learned support (Phi_pos, a diagonal Gaussian over trigram-hash
features of what that brick has already seen) beats the shared background (Phi_neg, fit from every
line ever seen); if NOTHING existing fits, a brand-new brick is spawned for it. Every piece here was
proven before being assembled: the quinary residual-carry update (exact, zero-tolerance conservation
check, native_quinary_debugging_arc.md), the gate (94.2% agreement with oracle routing, 6.871 vs
6.906 loss, lsl_gate_routing_test.py), and the win this exists to capture -- content-clustered
trained bricks beat a single D=384 net decisively on a fair same-corpus test (6.0-6.6 vs 10.3-11.7,
~1/7.6th the parameters). The ONE genuinely new, not-yet-separately-tested piece is ONLINE growth
(cluster_bricks_test.py only ever used a fixed, pre-known K=8); this file is that live test.
Still lossless at the single-brick level (same proven residual-carry mechanism, unchanged); the
routing decision itself is the one piece that is a real, honest approximation (a fitted gate, not a
proof) rather than an exact invariant -- same honesty standard the quinary conservation check has
always held itself to.
"""
import json, os, shutil, time, sys, argparse, numpy as np
from collections import defaultdict
from brick_cluster_charlm import BrickCluster, line_features

ARCH = "brick_cluster_quinary_v1"
BATCH_SIZE = 128
LR_SCALE = 1.0   # unchanged from the single-net path -- same residual-carry mechanism, same scale.
D_BRICK = 40     # ~10k params/brick (4*D^2 + 2*V*D at D=40, V~49 -> ~10184), matching the config
                 # content-clustered bricks were proven against in native_quinary_debugging_arc.md.

STATE = r"C:/galatea_gtlm/lutgtlm/mind.npz"
# FIXED vocab so the mind is stable across every ingestion (chars outside it are dropped).
VOCAB = "\n abcdefghijklmnopqrstuvwxyz0123456789.,!?;:'\"-()"
STOI = {c: i for i, c in enumerate(VOCAB)}; ITOS = {i: c for c, i in STOI.items()}; V = len(VOCAB)

enc = lambda s: [STOI[c] for c in s.lower() if c in STOI]


def _backup_incompatible(path, reason):
    ts = time.strftime("%Y%m%d_%H%M%S")
    bak = path.replace(".npz", f".backup_{ts}_pre_{reason}.npz")
    shutil.copy2(path, bak)
    print(f"[mind] existing mind is incompatible ({reason}) -- backed up (NEVER deleted) -> {bak}")
    return bak


def load_or_init():
    if os.path.exists(STATE):
        z = np.load(STATE)
        old_arch = str(z["ARCH"]) if "ARCH" in z else "pre_arch_marker"
        if old_arch != ARCH:
            # Different weight REPRESENTATION entirely (single D=384 net vs a growable cluster of
            # D=40 bricks) -- cannot reload, not just a width mismatch. Preserve the old mind (real
            # accumulated learning, never discard it) exactly like every prior architecture change.
            _backup_incompatible(STATE, f"arch_{old_arch}_to_{ARCH}")
            return BrickCluster(V, D_BRICK, seed=5), 0, 0
        cluster = BrickCluster.load_state(z)
        return cluster, int(z["seen"]), int(z["ingests"])
    return BrickCluster(V, D_BRICK, seed=5), 0, 0


def save(cluster, seen, ingests):
    dirpath = os.path.dirname(STATE)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    state = cluster.export_state()
    np.savez(STATE, ARCH=ARCH, seen=seen, ingests=ingests, **state)


def mean_ce(cluster, sample):
    """Held-out CE, routed through the REAL gate (not told which brick is 'correct') -- a genuine
    generalization test, not an oracle one. A line the gate can't route to any brick (empty
    cluster) is skipped, same as the old file skipped too-short lines."""
    tot, n = 0.0, 0
    for ln in sample:
        ids = enc(ln)[:121]
        if len(ids) < 3:
            continue
        logits = cluster.forward_logits(ids, line_features(ln))
        if logits is None:
            continue
        for t in range(len(ids) - 1):
            lg = logits[t]
            p = np.exp(lg - lg.max()); p /= p.sum()
            tot += -np.log(p[ids[t + 1]] + 1e-9); n += 1
    return tot / max(1, n)

def extract_text(path):
    """A .jsonl of {messages:[{role,content}]} (or {prompt,response}) -> 'role: content' lines; else raw text."""
    if path.lower().endswith(".jsonl"):
        lines = []
        for l in open(path, encoding="utf-8"):
            l = l.strip()
            if not l: continue
            try: d = json.loads(l)
            except Exception: continue
            if isinstance(d.get("messages"), list):
                for mm in d["messages"]:
                    r = mm.get("role", ""); c = str(mm.get("content", "")).strip()
                    if c: lines.append(f"{r}: {c}")
            else:
                for k in ("prompt", "question", "chosen", "response", "content", "text", "answer"):
                    if d.get(k): lines.append(str(d[k]).strip())
        return "\n".join(lines)
    return open(path, encoding="utf-8").read()


def _line_batches(ln):
    ids = enc(ln)
    if len(ids) < 3:
        return []
    buckets = defaultdict(list)
    for i in range(0, len(ids), 120):          # windows so long turns stay tractable
        w = ids[i:i + 121]
        if len(w) >= 3:
            buckets[len(w)].append(w)
    batches = []
    for wlen, windows in buckets.items():
        for i in range(0, len(windows), BATCH_SIZE):
            batches.append(windows[i:i + BATCH_SIZE])
    return batches


def ingest(cluster, text, passes, seen, on_round=None, held_out_seed=0):
    """BUG FIXED (2026-09-18, carried forward): held-out is a RANDOM ~10% sample (seeded,
    reproducible), not the tail.

    on_round(round_idx, passes, train_ce, ce, seen, n_bricks, elapsed) called after EVERY round --
    a long multi-pass run that gets killed or crashes must not lose everything already learned."""
    import random
    chunks = [ln for ln in text.split("\n") if ln.strip()]
    if len(chunks) < 4:
        held_out, train_lines = [], chunks
    else:
        k = min(40, max(1, len(chunks) // 10))
        rng = random.Random(held_out_seed)
        idx = set(rng.sample(range(len(chunks)), k))
        held_out = [c for i, c in enumerate(chunks) if i in idx]
        train_lines = [c for i, c in enumerate(chunks) if i not in idx]

    for r in range(1, passes + 1):
        t0 = time.time()
        order = list(range(len(train_lines)))
        random.Random(200 + r).shuffle(order)

        tot = 0.0; n = 0
        for li in order:
            ln = train_lines[li]
            batches = _line_batches(ln)
            if not batches:
                continue
            feats = line_features(ln)
            idx, line_loss = cluster.learn_line(batches, feats, lr_scale=LR_SCALE)
            n_windows = sum(len(b) for b in batches)
            if line_loss is not None:
                tot += line_loss * n_windows
            n += n_windows
            seen += n_windows
        train_ce = (tot / max(1, n)) if n else None

        # periodic re-merge, once per pass (not every line) -- same "periodic, not continuous"
        # cadence as Sovereign Core's own sleep-DAG, closer to k-means's global reassignment than
        # the one-shot greedy growth decision alone can be. See BrickCluster.consolidate().
        n_merged = cluster.consolidate()

        ce = mean_ce(cluster, held_out) if held_out else None
        elapsed = time.time() - t0
        if on_round is not None:
            on_round(r, passes, train_ce, ce, seen, len(cluster.bricks), n_merged, elapsed)

    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--say", default=None)
    a = ap.parse_args()
    cluster, seen, ingests = load_or_init()
    print(f"[mind] loaded: {seen} chars learned over {ingests} ingestions, {len(cluster.bricks)} "
          f"bricks, arch={ARCH}" if seen else f"[mind] fresh mind, arch={ARCH}")

    if a.say is not None:
        out = cluster.generate(enc(a.say), ITOS, line_features(a.say), n=60)
        print(f"  say {a.say!r:14s} -> {out!r}")
        return
    if not a.source:
        print("give a source file (.jsonl or .txt) to ingest, or --say 'seed' to sample.")
        return

    text = extract_text(a.source)
    print(f"[ingest] {a.source}: {len(text)} chars, {a.passes} passes ...")

    def on_round(r, passes, train_ce, ce, seen, n_bricks, n_merged, elapsed):
        train_str = f"{train_ce:.3f}" if train_ce is not None else "n/a"
        ce_str = f"{ce:.3f}" if ce is not None else "n/a"
        # INCREMENTAL SAVE every round -- a long multi-pass run that gets killed, crashes, or
        # just takes longer than expected must not lose everything already learned.
        save(cluster, seen, ingests_so_far[0])
        print(f"    round {r:2d}/{passes}  train_ce={train_str}  held_out_ce={ce_str}  "
              f"seen={seen}  bricks={n_bricks} (-{n_merged} merged)  ({elapsed:.1f}s)  [saved]", flush=True)

    ingests_so_far = [ingests + 1]   # mutable cell so on_round sees the post-increment count
    seen = ingest(cluster, text, a.passes, seen, on_round=on_round)
    ingests = ingests_so_far[0]
    save(cluster, seen, ingests)
    print(f"[mind] learned. now {seen} chars over {ingests} ingestions, {len(cluster.bricks)} bricks -> saved {STATE}")
    print(f"  sample  'i am' -> {cluster.generate(enc('i am'), ITOS, line_features('i am'), n=60)!r}")

if __name__ == "__main__":
    main()
