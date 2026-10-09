#!/usr/bin/env python3
r"""Real test of the question Jonathan posed: does LSL-style GATED routing (fit from data, used
at inference) hold up as well as the ORACLE k-means routing used in cluster_bricks_test.py's
6.0-6.6 result, or does that result only work because we cheated and told the system the right
answer in advance?

Design, matched to LSL's own algorithm (arXiv:2610.02126) as closely as this substrate allows:
- TRAINING is unchanged from cluster_bricks_test.py: oracle k-means still decides which lines each
  brick trains on. This matches LSL itself -- Algorithm 4 (FitAdapters) assumes IID access to the
  CURRENT phase's own data during training; the gate is never involved in training, only inference.
- The gate (Phi_pos per brick, Phi_neg shared) is fit on the SAME hashed-trigram feature space
  used for the original k-means clustering -- this is the direct analog of LSL's shared, frozen,
  brick-independent activation space (their gates sit on a frozen base model's shared activations,
  common across every adapter; our bricks have no shared internal representation since each is an
  independently-trained net, so the trigram-feature space is the correct "representation all
  bricks can be judged in" substitute).
- At inference: for EVERY line, ignore its oracle label, score it against every brick's Phi_pos
  minus the shared Phi_neg (diagonal-covariance Gaussian log-density, LSL's own simplification --
  their ablation found performance insensitive to component count, so a single-component gate is
  the right first test, not a shortcut), route to the best-scoring brick, and evaluate FORWARD-ONLY
  loss (no further training) under that brick's FINAL, already-trained weights. Compare against the
  same forward-only loss under oracle routing, on the identical frozen weights -- isolates exactly
  one variable: does the routing decision itself hold up, with training held constant.
"""
import numpy as np, random, time
from collections import defaultdict
from cluster_bricks_test import FixedStepBrick, det_hash, line_features, kmeans, make_batches
import lutgtlm_ingest as LI


def fit_diag_gaussian(X, var_floor):
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    var = np.maximum(var, var_floor)
    return mean, var


def log_gaussian_density(x, mean, var):
    return -0.5 * np.sum((x - mean) ** 2 / var + np.log(2 * np.pi * var))


def eval_forward_loss(brick, batch):
    """Forward-only cross-entropy, same formula learn_batch uses, no weight update."""
    logits = brick._forward_batch(batch)  # (B, T, V)
    idxs_arr = np.array(batch, dtype=np.int64)
    B, T, V = logits.shape
    tot = 0.0
    n = 0
    for t in range(T - 1):
        lg = logits[:, t, :]
        tgt = idxs_arr[:, t + 1]
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        tot += float(-np.log(p[np.arange(B), tgt] + 1e-9).sum())
        n += B
    return tot / max(1, n)


def main():
    import sys
    K = 8
    passes = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    CORPUS_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
    D_BRICK = 40
    VAR_FLOOR_FRAC = 0.05  # floor = this fraction of the pooled global per-dim variance

    text = LI.extract_text("voice_corpus.txt")[:CORPUS_CHARS]
    lines = [ln for ln in text.split("\n") if ln.strip()]
    print(f"[setup] {len(lines)} lines, K={K}", flush=True)

    feats = np.array([line_features(ln) for ln in lines])
    assign = kmeans(feats, K, seed=5)

    clusters = defaultdict(list)
    cluster_feat_idx = defaultdict(list)
    for i, (ln, a) in enumerate(zip(lines, assign)):
        clusters[a].append(ln)
        cluster_feat_idx[a].append(i)
    for k in sorted(clusters):
        print(f"  cluster {k}: {len(clusters[k])} lines", flush=True)

    # ---- TRAINING: identical to cluster_bricks_test.py (oracle routing, unchanged) ----
    bricks = {}
    batches_by_cluster = {}
    for k, cl_lines in clusters.items():
        batches = make_batches(cl_lines)
        if not batches:
            continue
        bricks[k] = FixedStepBrick(LI.V, D_BRICK, seed=5 + k)
        batches_by_cluster[k] = batches

    t0 = time.time()
    for p in range(passes):
        for k, brick in bricks.items():
            batches = batches_by_cluster[k]
            random.Random(p * 100 + k).shuffle(batches)
            losses = [brick.learn_batch(b, lr_scale=1.0).mean() for b in batches]
        if p % 5 == 0 or p == passes - 1:
            print(f"  training pass {p}: ({time.time()-t0:.0f}s)", flush=True)
    print(f"[training] done, {time.time()-t0:.0f}s total", flush=True)

    # ---- GATE FITTING: Phi_pos per brick, Phi_neg shared, on trigram features ----
    global_var = feats.var(axis=0)
    var_floor = np.maximum(global_var * VAR_FLOOR_FRAC, 1e-6)
    phi_neg_mean, phi_neg_var = fit_diag_gaussian(feats, var_floor)
    phi_pos = {}
    for k in bricks:
        idx = cluster_feat_idx[k]
        X = feats[idx]
        if len(X) < 2:
            # too few points for a variance estimate -- fall back to the global floor only
            phi_pos[k] = (X.mean(axis=0), var_floor.copy())
        else:
            phi_pos[k] = fit_diag_gaussian(X, var_floor)
    print(f"[gate] fit {len(phi_pos)} Phi_pos gates + 1 shared Phi_neg", flush=True)

    # ---- ROUTING TEST: for every line, gate-route vs oracle-route, same frozen weights ----
    agree = 0
    total_lines = 0
    oracle_losses = []
    gated_losses = []
    oracle_weights = []
    gated_weights = []

    for i, ln in enumerate(lines):
        oracle_k = int(assign[i])
        if oracle_k not in bricks:
            continue
        x = feats[i]
        margins = {}
        for k in bricks:
            mpos = log_gaussian_density(x, *phi_pos[k])
            mneg = log_gaussian_density(x, phi_neg_mean, phi_neg_var)
            margins[k] = mpos - mneg
        gated_k = max(margins, key=margins.get)
        total_lines += 1
        agree += int(gated_k == oracle_k)

        line_batches = make_batches([ln])
        if not line_batches:
            continue
        n_windows = sum(len(b) for b in line_batches)

        oracle_loss = np.mean([eval_forward_loss(bricks[oracle_k], b) for b in line_batches])
        gated_loss = np.mean([eval_forward_loss(bricks[gated_k], b) for b in line_batches])

        oracle_losses.append(oracle_loss); oracle_weights.append(n_windows)
        gated_losses.append(gated_loss); gated_weights.append(n_windows)

    oracle_weighted = float(np.average(oracle_losses, weights=oracle_weights))
    gated_weighted = float(np.average(gated_losses, weights=gated_weights))

    print(f"\n[RESULT] routing agreement: {agree}/{total_lines} lines ({100*agree/total_lines:.1f}%)")
    print(f"[RESULT] oracle-routed weighted mean forward loss: {oracle_weighted:.3f}")
    print(f"[RESULT] gated-routed  weighted mean forward loss: {gated_weighted:.3f}")
    print(f"[RESULT] gap (gated - oracle): {gated_weighted - oracle_weighted:+.3f}")


if __name__ == "__main__":
    main()
