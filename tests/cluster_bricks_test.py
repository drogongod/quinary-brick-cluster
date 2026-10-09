#!/usr/bin/env python3
r"""Tests Jonathan's "content-addressed/superposed bricks" proposal (2026-10-08, the session after
native_quinary_debugging_arc.md and galatea_fly_brick_cluster.md): instead of ONE D=384 net trying
to fit the whole 100KB corpus (ceiling: 5.3-6.7), split the corpus into K content-clusters and give
each cluster its own small, FULLY TRAINABLE (not frozen -- that was the already-disproven variant)
~10k-parameter brick. Hypothesis: a brick only has to fit its own narrow slice of the distribution,
which is an easier target than one net fitting everything -- so weighted-average loss across bricks
should beat the single-net ceiling even though each brick is individually far smaller.

Each brick reuses native_quinary_charlm.py's EXACT proven mechanism (same forward/backward, same
residual-carry, same L1 adaptive-normalize) but with the FIXED +-1-step formula (the best-tested
config from the debugging arc -- NOT the variable-step formula currently in the production file,
which measured worse).

Routing is IDEAL here (lines are clustered by content before training, each brick only ever sees
its own cluster) -- this isolates the core question (does narrower-per-brick data help convergence
at all) from the separate, unsolved engineering question of a real inference-time content router.
"""
import numpy as np, time, random
from collections import defaultdict
from native_quinary_charlm import NativeQuinaryCharLM, THRESHOLD
import lutgtlm_ingest as LI


class FixedStepBrick(NativeQuinaryCharLM):
    """Same class, but _residual_update_level uses the FIXED +-1-level-per-call step (the
    best-tested config: L1 normalize + fixed step, ceiling 5.3-6.7) instead of the variable-step
    formula now in the production file (measured worse, 9.0-11.0)."""
    def _residual_update_level(self, name, level, grad):
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled
        self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
        step = np.where(np.abs(res) >= THRESHOLD, np.sign(res), 0)
        desired = level.astype(np.int64) - step
        new_level = np.clip(desired, -2, 2).astype(np.int8)
        actual_step = level.astype(np.int64) - new_level.astype(np.int64)
        res = res - actual_step * THRESHOLD
        self._total_consumed[name] = self._total_consumed.get(name, 0) + int((actual_step * THRESHOLD).sum())
        self._res[name] = res
        return new_level


def det_hash(s):
    """Deterministic string hash (Python's builtin hash() is salted per-process) -- needed so
    clustering is reproducible run to run."""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def line_features(line, dim=64):
    v = np.zeros(dim)
    s = line.lower()
    for i in range(len(s) - 2):
        v[det_hash(s[i:i + 3]) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def kmeans(X, K, iters=30, seed=5):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), K, replace=False)].copy()
    assign = np.zeros(len(X), dtype=np.int64)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        assign = d.argmin(1)
        for k in range(K):
            pts = X[assign == k]
            if len(pts) > 0:
                centers[k] = pts.mean(0)
    return assign


def make_batches(lines):
    buckets = defaultdict(list)
    for ln in lines:
        ids = LI.enc(ln)
        if len(ids) < 3:
            continue
        for i in range(0, len(ids), 120):
            w = ids[i:i + 121]
            if len(w) >= 3:
                buckets[len(w)].append(w)
    batches = []
    for wlen, windows in buckets.items():
        for i in range(0, len(windows), 128):
            batches.append(windows[i:i + 128])
    return batches


def main():
    import sys
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    passes = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    CORPUS_CHARS = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    D_BRICK = 40   # ~10k params: 4*D^2 + 2*V*D at D=40, V~46 -> ~10080

    text = LI.extract_text("voice_corpus.txt")[:CORPUS_CHARS]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"[setup] {len(lines)} lines total, clustering into K={K} content-bricks, D_brick={D_BRICK}", flush=True)

    feats = np.array([line_features(ln) for ln in lines])
    assign = kmeans(feats, K, seed=5)

    clusters = defaultdict(list)
    for ln, a in zip(lines, assign):
        clusters[a].append(ln)
    for k in sorted(clusters):
        print(f"  cluster {k}: {len(clusters[k])} lines", flush=True)

    bricks = {}
    batches_by_cluster = {}
    weight_by_cluster = {}
    for k, cl_lines in clusters.items():
        batches = make_batches(cl_lines)
        if not batches:
            continue
        bricks[k] = FixedStepBrick(LI.V, D_BRICK, seed=5 + k)
        batches_by_cluster[k] = batches
        weight_by_cluster[k] = sum(len(w) for b in batches for w in b)

    total_weight = sum(weight_by_cluster.values())
    print(f"[setup] {len(bricks)} active bricks, total params ~{len(bricks) * D_BRICK * D_BRICK * 4 + 2*LI.V*D_BRICK*len(bricks)}", flush=True)

    t0 = time.time()
    per_brick_pass_loss = {k: [] for k in bricks}
    for p in range(passes):
        for k, brick in bricks.items():
            batches = batches_by_cluster[k]
            random.Random(p * 100 + k).shuffle(batches)
            losses = [brick.learn_batch(b, lr_scale=1.0).mean() for b in batches]
            per_brick_pass_loss[k].append(float(np.mean(losses)))
        weighted = sum(per_brick_pass_loss[k][-1] * weight_by_cluster[k] for k in bricks) / total_weight
        per_k_str = "  ".join(f"c{k}={per_brick_pass_loss[k][-1]:.2f}" for k in sorted(bricks))
        print(f"pass {p}: weighted_mean={weighted:.3f}  [{per_k_str}]  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
