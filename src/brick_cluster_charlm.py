#!/usr/bin/env python3
r"""BrickCluster -- the production substrate: many small, fully-trained quinary bricks instead of
one large net, routed by a real LSL-style density gate (not an oracle). Every piece here is exactly
the mechanism already proven this session, assembled for ongoing/online use:

  - FixedStepBrick: native_quinary_charlm.py's proven residual-carry mechanism (exactly, lossless)
    with L1 adaptive-normalize + FIXED +-1 step -- the best-tested single-brick config
    (native_quinary_debugging_arc.md), not the variable-step formula that measured worse.
  - Routing: an LSL-style (arXiv:2610.02126) density gate -- diagonal-covariance Gaussian Phi_pos
    per brick (fit from the trigram-hash features of the lines that brick has actually trained on)
    vs ONE shared Phi_neg (fit from every line ever seen, any brick) -- confirmed on a held-out
    oracle-vs-gated test to recover routing performance indistinguishable from, in fact slightly
    better than, knowing the right answer in advance (94.2% agreement, 6.871 vs 6.906 loss; see
    lsl_gate_routing_test.py / native_quinary_debugging_arc.md). Maintained ONLINE (incremental
    mean/sum-of-squares, exact, no refit-from-scratch needed) since production ingestion is
    streaming, not a fixed corpus known upfront -- the one piece genuinely new to this port.
  - Growth: a line is routed to the best-margin existing brick if ANY brick's Phi_pos beats the
    shared Phi_neg on it; otherwise nothing existing fits and a brand-new brick is spawned for it.
    This is the real, previously-flagged-as-untested piece (cluster_bricks_test.py only ever used a
    FIXED, pre-known K=8 -- this is the first test of genuine online growth).

Content-clustered trained bricks beat a single D=384 net decisively on a fair, same-corpus test
(6.0-6.6 vs 10.3-11.7, ~1/7.6th the parameters) -- see native_quinary_debugging_arc.md. This module
is that result, made continuously growable instead of a fixed one-shot K.
"""
import numpy as np
from native_quinary_charlm import (
    NativeQuinaryCharLM, THRESHOLD, level_to_planes, simd_mm, activation_quant_int8, sig_stable,
    EQ, MAX_EMB, EMB_DECAY_DIV, ACCUM_GRAD_CLIP, DH_CARRY_CLIP, BIAS_CEIL, BIAS_CORROB_ENABLED,
)

MATRICES = ["Win", "Wh", "Wgx", "Wgh", "Wo"]


class FixedStepBrick(NativeQuinaryCharLM):
    """L1-normalize + fixed +-1-level-per-call step -- the best-tested single-brick config
    (native_quinary_debugging_arc.md), not the production file's current variable-step formula.

    WIRED 2026-11-02: learn_batch below now commits flips via the validated-subset mechanism
    (ranked-candidate real-loss-checked accept/reject, Wemb included in the same pool, uncapped
    budget), ported from staggered_flip_test_v3.py's StaggeredFlipBrickV2(POLICY="validated_wemb").
    Root cause this replaces: the inherited NativeQuinaryCharLM.learn_batch commits every proposed
    flip from a single call simultaneously, measured to cost a +2 to +7 nat destructive interaction
    between flips that are each individually correct ("flip-avalanche," discrete_endpoint_audit.py).
    The test scripts validated this against one hardcoded fact string; this production version
    generalizes that to THIS CALL's own real batch (whatever idxs_batch actually is) as the
    validation signal -- the correct generalization, since every real training call already knows
    exactly what it is teaching. Measured result on the original single-fact test: exact, byte-for-
    byte correct generation ("The capital of France is Paris.") from rep 8 onward, floor 0.002-0.004
    nats, with NO float shadow weight anywhere -- see
    lucy_west_validated_subset_exact_generation_breakthrough_20261101.md.
    CHEAPER SELECTION (staggered_flip_test_v4.py/_v5_multifact.py): top-down bisection is NEGATIVE
    (unbounded-bad, 3.2x MORE total passes once failures get common); galloping/doubling search only
    wins a brick's very first-ever fact, then runs 0.4-0.8x MORE expensive than brute force with
    WORSE quality on every fact after -- see lucy_west_galloping_multifact_curriculum_negative_
    20261101.md. As of 2026-11-02, uncapped brute-force validation (VALIDATED_TOP_N=None below) is
    the only version that has actually been measured to work on more than one fact; this is a real,
    accepted compute-cost tradeoff (one extra forward pass over this call's batch per real candidate
    flip evaluated), not an oversight.
    CONFIRMED positive via the REAL router (staggered_flip_test_v7_real_router.py): an earlier 8-fact
    test's catastrophic forgetting was a capacity-allocation artifact of bypassing routing with one
    forced brick, not a defect in this update rule -- through the real, unmodified
    BrickCluster.route()/learn_line(), the same 8 facts auto-split into 3 bricks with NO catastrophic
    forgetting anywhere -- see lucy_west_real_router_prevents_collapse_20261101.md.
    CROSS-CATEGORY follow-up (staggered_flip_test_v8_cross_category.py): a second, different-subject
    passage shared a brick with an unrelated fact instead of spawning its own. Confirmed cold-start-
    only (corpus too small for IDF_MIN_DOCS to ever activate, not a standing router bug) -- see
    lucy_west_cross_category_why_shared_brick_mechanism_20261101.md and
    lucy_west_singleton_leniency_coldstart_only_confirmed_20261101.md. No change needed there; this
    class's own job is strictly the update rule, which the router correction does not touch."""

    VALIDATED_TOP_N = None  # None = uncapped (full real candidate pool each call) -- the only
    # budget actually measured to reach exact correct generation at a real floor (rep 8, 0.002-0.004
    # nats). Every capped budget tested (150/400/600/1000) was strictly worse and slower to
    # converge, with no sign of diminishing returns yet found at the top end -- see
    # lucy_west_validated_subset_exact_generation_breakthrough_20261101.md's "uncapped budget"
    # section. Set to an int here only if real production compute cost forces a budget; as of
    # 2026-11-02 that tradeoff has not been needed.

    def _residual_update_level(self, name, level, grad):
        """Kept standalone (unused by learn_batch below, which commits validated-subset flips
        itself) for any direct caller/diagnostic that still wants the plain fixed-step formula on
        its own, e.g. a one-off repair script."""
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

    def _batch_nll(self, idxs_arr):
        """Real teacher-forced mean NLL over THIS call's actual batch, evaluated at self's CURRENT
        (possibly mid-trial) weights. This is validated-subset's validation signal, generalized from
        the test scripts' single hardcoded FACT string to whatever this call is really teaching --
        every production training call already has exactly that information in idxs_batch."""
        logits = self._forward_batch(idxs_arr.tolist())  # (B, T, V) at current weights
        return self._nll_from_logits(logits, idxs_arr)

    @staticmethod
    def _nll_from_logits(logits, idxs_arr):
        """Shared by _batch_nll and the Wo-cache fast path below -- same math, same shape
        convention (B,T,V), so both give bit-identical loss for the same effective logits."""
        targets = idxs_arr[:, 1:]
        lg = logits[:, :-1, :]
        logp = lg - np.logaddexp.reduce(lg, axis=-1, keepdims=True)
        b_idx = np.arange(targets.shape[0])[:, None]
        t_idx = np.arange(targets.shape[1])[None, :]
        nll = -logp[b_idx, t_idx, targets]
        return float(nll.mean())

    def _wo_only_nll(self, Hq, Hs, idxs_arr):
        """Fast path for validated-subset candidates that only touch Wo: Wo maps an already-
        computed hidden state to logits and never feeds back into the recurrence (unlike
        Win/Wh/Wgx/Wgh/Wemb, which all change h_t itself), so re-evaluating a Wo-only trial
        never needs to redo the T-step recurrence -- only the output projection, against hidden
        states cached from the last time the recurrence was actually run. Measured exact (not
        approximate): same loss value, diff 0.00e+00, 4.92x faster per call at this call's real
        candidate pool (lucy_west_... Wo-cache benchmark, 2026-11-02). MUST be invalidated
        (recomputed via _cache_hidden_quant) any time a non-Wo candidate actually commits."""
        T = idxs_arr.shape[1]
        S1_o, S2_o = level_to_planes(self.Wo)
        logits = np.empty((T, self.V, idxs_arr.shape[0]), np.float64)
        for t in range(T):
            logits[t] = simd_mm(S1_o, S2_o, Hq[t], Hs[t])
        return self._nll_from_logits(logits.transpose(2, 0, 1), idxs_arr)

    def _cache_hidden_quant(self, idxs_arr):
        """Forward-only (no logits, no backward) recompute of the quantized hidden-state
        sequence at self's CURRENT Win/Wh/Wgx/Wgh/Wemb -- what _wo_only_nll reads from. Only
        called when the cache is invalidated (a non-Wo candidate actually committed); the
        INITIAL cache per learn_batch call is free (reused directly from H, computed by the
        forward pass every call already does for the gradient)."""
        B, T = idxs_arr.shape[0], idxs_arr.shape[1]
        D = self.D
        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        X = (self.Wemb[:, idxs_arr.T]).astype(np.float64) / EQ
        h = np.zeros((D, B), np.float64)
        Hq, Hs = [], []
        for t in range(T):
            x_t = X[:, t, :]
            xq, xs = activation_quant_int8(x_t); hq, hs = activation_quant_int8(h)
            pc = simd_mm(S1_in, S2_in, xq, xs) + simd_mm(S1_h, S2_h, hq, hs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_gx, S2_gx, xq, xs) + simd_mm(S1_gh, S2_gh, hq, hs) + self.bg[:, None]
            g = sig_stable(pg)
            h = g * c + (1 - g) * h
            hnq, hns = activation_quant_int8(h)
            Hq.append(hnq); Hs.append(hns)
        return Hq, Hs

    def learn_batch(self, idxs_batch, lr_scale=1.0, corrob_eligible=True):
        self.consume_pending_relay()
        B = len(idxs_batch); T = len(idxs_batch[0])
        idxs_arr = np.array(idxs_batch, dtype=np.int64)
        D = self.D

        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        S1_o, S2_o = level_to_planes(self.Wo)
        S1_inT, S2_inT = level_to_planes(self.Win.T); S1_hT, S2_hT = level_to_planes(self.Wh.T)
        S1_gxT, S2_gxT = level_to_planes(self.Wgx.T); S1_ghT, S2_ghT = level_to_planes(self.Wgh.T)
        S1_oT, S2_oT = level_to_planes(self.Wo.T)

        X = (self.Wemb[:, idxs_arr.T]).astype(np.float64) / EQ
        h = np.zeros((D, B), np.float64)
        H = []
        logits = np.empty((T, self.V, B), np.float64)
        for t in range(T):
            x_t = X[:, t, :]
            xq, xs = activation_quant_int8(x_t); hq, hs = activation_quant_int8(h)
            pc = simd_mm(S1_in, S2_in, xq, xs) + simd_mm(S1_h, S2_h, hq, hs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_gx, S2_gx, xq, xs) + simd_mm(S1_gh, S2_gh, hq, hs) + self.bg[:, None]
            g = sig_stable(pg)
            hn = g * c + (1 - g) * h
            H.append((x_t, h, c, g, hn)); h = hn
            hnq, hns = activation_quant_int8(hn)
            logits[t] = simd_mm(S1_o, S2_o, hnq, hns)

        gWemb = np.zeros_like(self.Wemb, dtype=np.float64)
        gWin = np.zeros((D, D)); gWh = np.zeros((D, D)); gb = np.zeros(D)
        gWgx = np.zeros((D, D)); gWgh = np.zeros((D, D)); gbg = np.zeros(D)
        gWo = np.zeros((self.V, D))
        dh = np.zeros((D, B)); loss = np.zeros(B); n = 0

        for t in range(T - 2, -1, -1):
            tgt = idxs_arr[:, t + 1]
            lg = logits[t]
            p = np.exp(lg - lg.max(axis=0, keepdims=True)); p /= p.sum(axis=0, keepdims=True)
            loss += -np.log(p[tgt, np.arange(B)] + 1e-9); n += 1
            dl = p.copy(); dl[tgt, np.arange(B)] -= 1.0
            x_t, hp, c, g, hn = H[t]
            gWo += dl @ hn.T
            dlq, dls = activation_quant_int8(dl)
            dh = np.clip(dh + simd_mm(S1_oT, S2_oT, dlq, dls), -DH_CARRY_CLIP, DH_CARRY_CLIP)
            dg = dh * (c - hp); dc = dh * g; dhp = dh * (1 - g)
            dpc = dc * (1 - c * c); dpg = dg * g * (1 - g)
            gWin += dpc @ x_t.T; gWh += dpc @ hp.T; gb += dpc.sum(axis=1)
            gWgx += dpg @ x_t.T; gWgh += dpg @ hp.T; gbg += dpg.sum(axis=1)
            dpcq, dpcs = activation_quant_int8(dpc); dpgq, dpgs = activation_quant_int8(dpg)
            dxemb = simd_mm(S1_inT, S2_inT, dpcq, dpcs) + simd_mm(S1_gxT, S2_gxT, dpgq, dpgs)
            np.add.at(gWemb, (slice(None), idxs_arr[:, t]), dxemb)
            dh = np.clip(dhp + simd_mm(S1_hT, S2_hT, dpcq, dpcs) + simd_mm(S1_ghT, S2_ghT, dpgq, dpgs),
                         -DH_CARRY_CLIP, DH_CARRY_CLIP)

        scale_int = lambda G: np.clip(np.round(G * lr_scale), -ACCUM_GRAD_CLIP, ACCUM_GRAD_CLIP).astype(np.int64)
        grads = {"Win": gWin, "Wh": gWh, "Wgx": gWgx, "Wgh": gWgh, "Wo": gWo}

        proposals = {}
        for name in MATRICES:
            level = getattr(self, name)
            mom_int = self._integer_ema(name, scale_int(grads[name]))
            scaled = self._adaptive_normalize(name, mom_int)
            res_prev = self._res.get(name, np.zeros_like(level, dtype=np.int64))
            res_new = res_prev + scaled
            self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
            step = np.where(np.abs(res_new) >= THRESHOLD, np.sign(res_new), 0)
            proposals[name] = (level.copy(), res_new, step)

        wemb_grad_int = scale_int(gWemb * EQ)
        wmom_int = self._integer_ema("Wemb", wemb_grad_int)
        wscaled = self._adaptive_normalize_v2("Wemb", wmom_int)
        wres_prev = self._res.get("Wemb", np.zeros_like(self.Wemb, dtype=np.int64))
        wres_new = wres_prev + wscaled
        self._total_in["Wemb"] = self._total_in.get("Wemb", 0) + int(wscaled.sum())
        wstep = np.sign(wres_new) * (np.abs(wres_new) // THRESHOLD)
        wemb_prop = (self.Wemb.copy(), wres_new, wstep)

        candidates = []
        for name, (level, res_new, step) in proposals.items():
            for ij in np.argwhere(step != 0):
                ij = tuple(ij)
                candidates.append((name, ij, abs(int(res_new[ij])), int(step[ij])))
        wlevel, wres_new2, wstep2 = wemb_prop
        for ij in np.argwhere(wstep2 != 0):
            ij = tuple(ij)
            candidates.append(("Wemb", ij, abs(int(wres_new2[ij])), int(wstep2[ij])))
        candidates.sort(key=lambda c: -c[2])

        commit_set = set()
        eval_pool = candidates if self.VALIDATED_TOP_N is None else candidates[:self.VALIDATED_TOP_N]
        trial_levels = {name: proposals[name][0].copy() for name in MATRICES}
        trial_wemb = wemb_prop[0].copy()
        cur_loss = self._batch_nll(idxs_arr)
        # Wo-cache: H's hn (post-step hidden state) is already sitting in memory from the
        # forward pass above, free to requantize -- this is the SAME state _batch_nll's
        # _forward_batch would have produced just now, so the cache starts valid.
        _hq_hs = [activation_quant_int8(H[t][4]) for t in range(T)]
        Hq_cache = [q for q, s in _hq_hs]; Hs_cache = [s for q, s in _hq_hs]
        cache_valid = True
        for name, ij, mag, step_val in eval_pool:
            if name == "Wemb":
                old_val = int(trial_wemb[ij])
                trial_wemb[ij] = old_val - step_val
                self.Wemb = trial_wemb
            else:
                old_val = trial_levels[name][ij]
                trial_levels[name][ij] = int(np.clip(old_val - step_val, -2, 2))
                setattr(self, name, trial_levels[name])

            if name == "Wo":
                if not cache_valid:
                    Hq_cache, Hs_cache = self._cache_hidden_quant(idxs_arr)
                    cache_valid = True
                new_loss = self._wo_only_nll(Hq_cache, Hs_cache, idxs_arr)
            else:
                new_loss = self._batch_nll(idxs_arr)

            if new_loss <= cur_loss + 1e-9:
                commit_set.add((name, ij))
                cur_loss = new_loss
                if name != "Wo":
                    cache_valid = False  # h_t just changed; next Wo eval must recompute first
            else:
                if name == "Wemb":
                    trial_wemb[ij] = old_val
                    self.Wemb = trial_wemb
                else:
                    trial_levels[name][ij] = old_val
                    setattr(self, name, trial_levels[name])

        for name in MATRICES:
            setattr(self, name, proposals[name][0])
        self.Wemb = wemb_prop[0]

        for name, (level, res_new, step) in proposals.items():
            mask = np.zeros_like(step, dtype=bool)
            for n2, ij in commit_set:
                if n2 == name:
                    mask[ij] = True
            actual_step = np.where(mask, step, 0)
            desired = level.astype(np.int64) - actual_step
            new_level = np.clip(desired, -2, 2).astype(np.int8)
            realized = level.astype(np.int64) - new_level.astype(np.int64)
            res_final = res_new - realized * THRESHOLD
            self._total_consumed[name] = self._total_consumed.get(name, 0) + int((realized * THRESHOLD).sum())
            self._res[name] = res_final
            setattr(self, name, new_level)

        mask = np.zeros_like(wstep2, dtype=bool)
        for n2, ij in commit_set:
            if n2 == "Wemb":
                mask[ij] = True
        realized_step = np.where(mask, wstep2, 0)
        W_mid = wlevel.astype(np.int64) - realized_step
        decay = np.sign(W_mid) * (np.abs(W_mid) // EMB_DECAY_DIV)
        W_new = W_mid - decay
        clipped = np.clip(W_new, -MAX_EMB, MAX_EMB)
        discarded = int(decay.sum()) + int((W_new - clipped).sum())
        self._total_discarded = getattr(self, "_total_discarded", {})
        self._total_discarded["Wemb"] = self._total_discarded.get("Wemb", 0) + discarded
        self._total_consumed["Wemb"] = self._total_consumed.get("Wemb", 0) + int((realized_step * THRESHOLD).sum())
        self._res["Wemb"] = wres_new2 - realized_step * THRESHOLD
        self.Wemb = clipped

        if BIAS_CORROB_ENABLED and corrob_eligible:
            new_b, self._b_streak_count, self._b_streak_sign = self._corroborated_bias_step(
                self.b, gb, lr_scale, self._b_streak_count, self._b_streak_sign)
            new_bg, self._bg_streak_count, self._bg_streak_sign = self._corroborated_bias_step(
                self.bg, gbg, lr_scale, self._bg_streak_count, self._bg_streak_sign)
        else:
            new_b = self.b - lr_scale * 1e-4 * gb
            new_bg = self.bg - lr_scale * 1e-4 * gbg
        self.b, self._last_excess_b = self._clip_and_shed(new_b, BIAS_CEIL)
        self.bg, self._last_excess_bg = self._clip_and_shed(new_bg, BIAS_CEIL)

        return loss / max(1, n)


# 🎯 2026-10-09, ROOT CAUSE #5 -- found by directly instrumenting consolidate()'s ACCEPTED merges
# (not just the clearly-rejected ones _diag_merge_mechanics.py had been printing) after the
# MERGE_SLACK re-sweep (0.0-3.0, then 10-5000) showed huge flat dead zones with only two narrow
# real transitions (10->15, 700->800) -- the real accepted-merge margins were not borderline at
# all, they were enormous and positive (+168 to +256 log-density units, see
# _diag_accepted_merges.py), meaning slack can never meaningfully gate them in any sane range.
# Traced the actual lines behind one such cascade (a single hub brick absorbing 16 different
# single-line bricks with huge positive margins, _diag_hub_lines.py): completely unrelated
# content -- "Cato the Younger", "Sylvia Plath's Apotheosis", "SolarWinds supply chain",
# "I am Galatea, the Philosopher-Queen" declarations -- being treated as near-identical. Measured
# directly why: real corpus lines average ~3000 trigrams (long paragraphs), hashed into only 64
# buckets -- ~46 collisions per bucket on AVERAGE, so literally every one of the 64 buckets is
# nonzero for literally every line regardless of content (confirmed on 5 real lines, 0 exceptions).
# TF-IDF (IDF_MIN_DOCS/IDF_SMOOTHING, the previous fix) downweights buckets that are common ACROSS
# lines, but cannot recover signal already destroyed WITHIN a single line by averaging ~46
# unrelated trigrams into one scalar per bucket before IDF ever sees it -- a strictly earlier,
# deeper capacity problem than the one TF-IDF fixed. Standard hashing-trick practice (e.g.
# scikit-learn's HashingVectorizer) uses dims in the thousands-to-millions for exactly this
# reason. 8192 brings average occupancy for a ~3000-trigram line down to ~0.37/bucket (most
# buckets 0 or 1 trigram, genuinely sparse) -- O(FEATURE_DIM) per Gaussian update/log_density is
# still trivial (microseconds) next to a brick's own forward/backward pass. Re-verify the hub-line
# and full sweep diagnostics after any change to this constant -- do not assume collision rate
# without measuring it against real line lengths.
FEATURE_DIM = 8192


def det_hash(s):
    """Deterministic string hash (Python's builtin hash() is salted per-process) -- reproducible
    across a save/load/restart cycle, unlike the builtin."""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def raw_trigram_sparse(line, dim=FEATURE_DIM):
    """dict {bucket: count} -- O(line_length) time, O(distinct_buckets) space, NEVER allocates a
    dense dim-sized array. 2026-10-09, added alongside FEATURE_DIM 64->8192 (see that constant's
    comment): raising dim fixed real hash-collision false-merges but made every DENSE per-line
    operation 128x more expensive; real lines are sparse (a few hundred to ~1200 active buckets
    out of 8192, see _diag_hub_lines.py measurements), so routing/updates should cost O(nnz), not
    O(dim). This is the hashing primitive the sparse hot path (tfidf_sparse, _OnlineGaussian's
    sparse update/log_density) is built on; raw_trigram_counts below is kept dense for backward
    compatibility with existing diagnostics that expect a dense array."""
    counts = {}
    s = line.lower()
    for i in range(len(s) - 2):
        b = det_hash(s[i:i + 3]) % dim
        counts[b] = counts.get(b, 0) + 1
    return counts


def raw_trigram_counts(line, dim=FEATURE_DIM):
    v = np.zeros(dim)
    for b, c in raw_trigram_sparse(line, dim).items():
        v[b] = c
    return v


def line_features(line, dim=FEATURE_DIM):
    """Plain L2-normalized raw counts, no IDF weighting -- kept for callers with no corpus-wide
    document-frequency context (standalone tests, one-off scripts). Production use goes through
    BrickCluster.tfidf_features instead; see its docstring for why raw counts alone are not
    enough at real scale (confirmed directly: 98% of real-corpus line pairs score >0.9 cosine
    similarity under raw counts despite being entirely unrelated content -- see
    brick_cluster_production_port.md)."""
    v = raw_trigram_counts(line, dim)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class _OnlineGaussian:
    """Exact incremental diagonal-covariance Gaussian -- count/sum/sum-of-squares per dimension,
    so mean/var are always exact (not an EMA approximation) and cost O(dim) per update, no refit.

    freeze_at_n (2026-10-09, real bug found running against the actual 4.2MB corpus): without a
    cap, a Gaussian fit from EVERY example ever seen keeps widening as more diverse content
    arrives, which -- for the SHARED background (Phi_neg) specifically -- creates a self-
    reinforcing runaway: wider Phi_neg inflates the adaptive variance floor on every brick's own
    Phi_pos (see mean_var below), which makes EVERY acceptance test more lenient, which merges
    more bricks together, which widens their Phi_pos further, accelerating collapse. Measured
    directly, same corpus, same MERGE_SLACK, only size varying: 70 bricks at 224 lines -> 49 at
    532 -> 6 at 819 -> 1 (total collapse) at 1257 and beyond -- a smooth degradation with SCALE,
    not a threshold effect, confirming the background itself was the thing drifting, not the
    merge constant. This also means the port had quietly deviated from the actual LSL paper:
    arXiv:2610.02126's own Phi_neg is explicitly "a SMALL sample from a generic reference
    dataset... does not require more than 1M tokens" -- a bounded, roughly-stationary reference,
    not an unboundedly growing one. freeze_at_n=None (the default) keeps the old unbounded
    behavior, correct for a brick's own Phi_pos (which SHOULD reflect everything it has actually
    absorbed); BrickCluster's shared `neg` is constructed with a real cap instead."""
    def __init__(self, dim=FEATURE_DIM, freeze_at_n=None):
        self.n = 0
        self.sum = np.zeros(dim)
        self.sumsq = np.zeros(dim)
        self.freeze_at_n = freeze_at_n
        # CACHE (2026-10-09, companion fix to raw_trigram_sparse above): mean/var/baseline only
        # actually change when THIS brick's own (n, sum, sumsq) change (via update() or a merge),
        # never just because some OTHER brick/line is being evaluated. Before this cache, route()
        # recomputed mean_var from scratch (a full O(dim) dense op) for EVERY brick on EVERY line
        # -- confirmed directly to dominate real runtime (round 1 of a 3M-char/1829-line corpus
        # took 377s; a 4.2M/3799-line round took 1700s, i.e. ~28 minutes). Keying the cache on
        # self.n is sufficient and safe: n changes IFF sum/sumsq change (every mutation path
        # increments n alongside them, never independently), so a stale cache is impossible.
        self._cache_n = -1
        self._cache_gvar_id = None
        self._cache_mean = None
        self._cache_var = None
        self._cache_baseline = None   # scalar: sum_i[mean_i^2/var_i + log(2*pi*var_i)], the exact
        # dense term log_density's sparse correction (see below) is computed relative to.

    def update(self, x):
        """x is (idx_array, val_array) [sparse] or a dense ndarray -- see _as_sparse. Sparse
        update costs O(nnz), not O(dim): self.sum/self.sumsq stay fully dense (required for exact
        merge-by-addition in BrickCluster.consolidate), but only the touched indices are written."""
        if self.freeze_at_n is not None and self.n >= self.freeze_at_n:
            return
        idx, val = _as_sparse(x)
        self.n += 1
        self.sum[idx] += val
        self.sumsq[idx] += val * val

    def _cached_stats(self, var_floor_frac, global_var):
        """Returns (mean, var, baseline), recomputing (all O(dim), done together) only on a real
        cache miss -- i.e. only for the ONE brick that just changed, not for every brick being
        compared against it."""
        if self.n == self._cache_n and id(global_var) == self._cache_gvar_id:
            return self._cache_mean, self._cache_var, self._cache_baseline
        mean, var = self._mean_var_uncached(var_floor_frac, global_var)
        baseline = float(np.sum(mean * mean / var + np.log(2 * np.pi * var)))
        self._cache_n = self.n
        self._cache_gvar_id = id(global_var)
        self._cache_mean, self._cache_var, self._cache_baseline = mean, var, baseline
        return mean, var, baseline

    def mean_var(self, var_floor_frac, global_var):
        """Kept as a public dense (mean, var) accessor for existing diagnostics -- backed by the
        same cache as log_density, so calling both doesn't duplicate the O(dim) recompute."""
        mean, var, _ = self._cached_stats(var_floor_frac, global_var)
        return mean, var

    def _mean_var_uncached(self, var_floor_frac, global_var):
        if self.n == 0:
            return np.zeros_like(self.sum), np.maximum(global_var * var_floor_frac, 1e-6)
        mean = self.sum / self.n
        var = np.maximum(self.sumsq / self.n - mean * mean, 0.0)
        # ADAPTIVE floor: a brick fit from very few samples genuinely doesn't know its own spread
        # yet -- flooring it at a small FIXED fraction of global variance makes a 1-2-sample brick
        # look falsely sharp/confident, rejecting the very next (possibly related) line and
        # spawning again. Scale the floor toward the FULL global variance when n is small (as wide
        # as the background -- "I don't know yet, don't reject on this basis") and relax toward the
        # base var_floor_frac as real evidence accumulates. Found via direct testing
        # (_test_brick_ingest.py): without this, 44 bricks spawned in a single pass and growth
        # stalled -- cold-start over-spawning, not a hypothetical concern.
        effective_floor_frac = max(var_floor_frac, 1.0 / max(self.n, 1))
        floor = np.maximum(global_var * effective_floor_frac, 1e-6)
        # CEILING (2026-10-09, second half of the same runaway bug -- freezing Phi_neg only fixed
        # one path to it). A brick's own REAL variance (line 108) had a floor but no ceiling: every
        # merge adds the absorbed brick's sum/sumsq directly in, so a brick that happens to absorb
        # a few early merges keeps genuinely widening without limit, which makes IT more likely to
        # accept the next merge and the next routed line (a wide Gaussian assigns real density to
        # almost anything) -- the identical self-reinforcing runaway as the Phi_neg bug, just
        # running through a brick's own accumulated stats instead of the shared background. Capping
        # at a multiple of the now-frozen global_var closes this second path the same way freezing
        # Phi_neg closed the first -- global_var is now a stable reference to cap against, not a
        # live-growing one, so this ceiling doesn't drift over time the way an uncapped one would.
        ceiling = global_var * VAR_CEILING_MULT
        return mean, np.clip(var, floor, ceiling)

    def log_density(self, x, var_floor_frac, global_var):
        # MEAN over dims, not SUM (2026-10-09, found raising FEATURE_DIM 64->8192 to fix the real
        # hash-collision bug): a SUM accumulates one term per dimension, so any small systematic
        # per-dim bias between two Gaussians (e.g. a young brick's inflated 1/n floor vs Phi_neg's
        # mature one) scales with dim count -- margins that were +-tens/hundreds at dim=64 became
        # +-thousands at dim=8192 purely from summing 128x more terms, silently invalidating every
        # previously-swept MERGE_SLACK value with no change in real similarity. Averaging makes the
        # margin's SCALE independent of FEATURE_DIM (routing's margin>0 test is unaffected either
        # way -- dividing by a positive constant never flips a sign -- only MERGE_SLACK's meaning
        # was at stake). Re-sweep MERGE_SLACK fresh at this scale; do not reuse old values.
        #
        # SPARSE CORRECTION (2026-10-09, companion fix): the mean-over-dims sum is additively
        # separable per dimension, so it splits EXACTLY (not approximately) into a per-brick
        # BASELINE (sum over ALL dims assuming x=0 everywhere, cached in _cached_stats -- O(dim),
        # but only recomputed when THIS brick's own stats change) plus a CORRECTION over just x's
        # nonzero entries (O(nnz)): for i where x_i=0, the term is mean_i^2/var_i + log(2pi var_i),
        # already folded into baseline; for i where x_i!=0, swap in the true term. Real lines are
        # sparse (a few hundred to ~1200 nonzero of 8192, _diag_hub_lines.py), so this turns
        # route()'s dominant O(bricks*lines*dim) cost into O(bricks*lines) cache-hits + O(nnz)
        # corrections -- confirmed exact, not approximate, via _diag_sparse_exactness.py before
        # shipping (bit-identical margins to the old dense computation on real data).
        mean, var, baseline = self._cached_stats(var_floor_frac, global_var)
        idx, val = _as_sparse(x)
        if idx.size == 0:
            correction = 0.0
        else:
            m = mean[idx]
            v = var[idx]
            correction = float(np.sum((val - m) ** 2 / v - m * m / v))
        dim = mean.shape[0]
        return -0.5 / dim * (baseline + correction)


def _as_sparse(x):
    """Normalize a feature vector to (idx_array, val_array) sparse form. x may already be sparse
    (a 2-tuple) or a dense ndarray (existing diagnostics pass dense arrays -- handled transparently
    by scanning its nonzeros, same result, just O(dim) to find them instead of O(nnz))."""
    if isinstance(x, tuple):
        return x
    idx = np.nonzero(x)[0]
    return idx, x[idx]


VAR_FLOOR_FRAC = 0.05

MAX_RELAY_HOPS = 3    # 2026-10-28, Jonathan's explicit spec for the dam/cups relay: "if it bounces
                       # more than two or three times... it will disintegrate" -- his own worked
                       # example (7 -> 4+3 -> 2+2,1+2 -> ...) converges to terminal/destroyed values
                       # within 2-3 splits, not via pure halving all the way to zero. Matters because
                       # pure halving-with-no-depth-limit is a REAL problem, not just slow: each hop
                       # can split into 2 fresh (never-before-visited) targets, so unlimited depth
                       # means up to 2**depth separate in-flight packets -- confirmed directly
                       # (repair_bias_runaway.py against the real backup checkpoint) that a
                       # historical ~244M-magnitude backlog needs ~25 halvings to fall under
                       # BIAS_CEIL, which would mean up to 2**25 (33 million) packets, computationally
                       # infeasible. Capping real relay hops at 3 bounds this to at most 2**3=8
                       # packets per original overflow event. Whatever excess remains after
                       # MAX_RELAY_HOPS is deliberately destroyed (see _relay_excess), not relayed
                       # further -- a bounded, honest exception for backlog this catastrophic, which
                       # only exists because BIAS_CEIL didn't exist yet when it accumulated; ordinary
                       # LIVE per-call overflow is small (bounded by one gradient step) and should
                       # resolve to zero excess within 1 hop almost always, never needing this path.

BIAS_CORROB_REUSE_THRESHOLD = 3   # 2026-10-29, real isolation finding (see BIAS_CORROB_ENABLED's
                       # comment in native_quinary_charlm.py for the full real multi-seed A/B that
                       # found this): corroboration-gating on the bias update only helps once a
                       # brick has been reactivated at least this many times BEFORE the current
                       # call -- below this, the mechanism almost never gets the second confirming
                       # call it needs, so it only delays corrections rather than ever actually
                       # smoothing them. Owned here, not in native_quinary_charlm.py, because only
                       # the cluster knows how many times a brick has been reactivated
                       # (brick_pos[idx].n) -- same separation already used for pending_relay's
                       # visited-history. Value 3 is the exact real-tested split from
                       # isolation_test_bias_corrob.py (clean ~10x-worse vs ~3x-better separation
                       # across all 3 seeds at this threshold), not a guess.

# 🛑 2026-10-09, real bug found running the actual 4.2MB production corpus (not a smaller test
# slice): the shared Phi_neg was growing from EVERY line ever seen, unboundedly. Measured
# directly, same corpus, same MERGE_SLACK=3.0, only size varying: 70 bricks at 224 lines -> 49 at
# 532 -> 6 at 819 -> 1 (total collapse) at 1257+ lines -- a smooth degradation with scale, not a
# threshold effect tied to the merge constant. Root cause: a wider Phi_neg inflates the adaptive
# variance floor on every brick's Phi_pos (see _OnlineGaussian.mean_var), which makes EVERY
# acceptance test more lenient as more total data flows through, a self-reinforcing runaway no
# fixed MERGE_SLACK value can fully prevent. This was a real deviation from the actual LSL paper
# (arXiv:2610.02126), whose own Phi_neg is explicitly "a SMALL sample... does not require more
# than 1M tokens" -- bounded on purpose, not grown forever. NEG_FREEZE_AT_N caps the shared
# background at a sample size already confirmed healthy (224 lines produced 70 well-formed
# bricks with good quality) with real margin above it.
NEG_FREEZE_AT_N = 300

# 🛑 2026-10-09, SECOND half of the same runaway bug, found by re-examining the mechanism after
# Jonathan flagged "signs of an underlying problem unfixed" even after NEG_FREEZE_AT_N -- he was
# right. Freezing Phi_neg closed ONE path to runaway acceptance (the shared background growing
# unboundedly); a brick's OWN variance had a floor (above) but no ceiling, so a brick that happens
# to absorb a few early merges keeps genuinely widening without limit as more merged content's
# sum/sumsq gets added in -- the identical self-reinforcing dynamic, running through a single
# brick's real stats instead of the shared background. VAR_CEILING_MULT caps any Gaussian's
# variance at this multiple of the (now-frozen, stable) global_var -- see _OnlineGaussian.mean_var.
VAR_CEILING_MULT = 4.0

# 🎯 2026-10-09, THE REAL ROOT CAUSE, found by directly instrumenting a merge pass on the actual
# 1.5M-char corpus after NEITHER the Phi_neg freeze NOR the variance ceiling changed the outcome
# at all. Measured directly: mean pairwise cosine similarity among 50 real, completely unrelated
# lines (WannaCry / Bangladesh Bank / DarkWeb market / Equifax / Target breach -- different
# incidents, different content) was 0.965, with 98% of all pairs scoring above 0.9. Raw bag-of-
# trigram-hash COUNTS over long (~3000-char) English prose is dominated by universally common
# patterns ("the", "ing", spaces, punctuation) that look nearly identical across any sufficiently
# long passage, regardless of topic -- topic-specific signal gets drowned out by sheer volume of
# generic trigrams. This was never a gate-logic bug: the gate was behaving correctly on features
# that had already lost almost all distinguishing signal before the gate ever saw them. Neither
# candidate-limiting, variance-capping, nor background-freezing could fix a problem one layer
# below the gate. Real, standard fix: TF-IDF -- downweight trigrams that appear in most lines
# (uninformative), upweight the ones specific to few (actually distinguishing). IDF_MIN_DOCS
# gates when a trigram bucket's IDF weight is trusted (too few documents seen = too noisy an
# estimate of how "common" it really is); IDF_SMOOTHING is the standard +1 Laplace term.
IDF_MIN_DOCS = 20
IDF_SMOOTHING = 1.0
# 🎯 2026-11-01, real cross-category test confirmed this threshold mattering directly: a 14-line
# test (lucy_west_cross_category_why_shared_brick_mechanism_20261101.md) never reached n_docs=20,
# so IDF weighting was OFF for its entire duration, and a growing narrative passage landed in an
# unrelated singleton brick (margin -0.0217 on the real 6-fact brick -- genuinely rejected there,
# margin +0.1048 on a freshly-spawned n=1 brick, which the adaptive variance floor below makes the
# most lenient brick in the cluster by construction). Not a router bug: real safeguards (this IDF
# gate, and the floor narrowing with n) simply never got enough data to engage at this scale.
# Testing the real category-resolution question needs >=20 real lines merging into EXPERIENCED
# bricks, not a handful of lines mostly landing in freshly-spawned singletons.
# 🔧 2026-11-01, four candidate fixes for the above tested against the real trace (Jonathan's ask:
# "make certain two unlike bricks do not occupy the same brick when all else is even"). Raising the
# route() spawn-margin bar is FALSIFIED: legitimate merges (margin as low as 0.0008) are as thin as
# or thinner than the spurious one (0.0237-0.1048) -- no threshold separates them. Lowering
# IDF_MIN_DOCS is FALSIFIED, makes it WORSE (<=2 collapses everything into one brick -- IDF from
# too few docs is itself unreliable). A lexical-overlap check against brick_taught_lines also fails
# -- legitimate merges share zero content words with each other, only the template. The one real
# working fix found: force _spawn() unconditionally while self.n_docs < IDF_MIN_DOCS (defer ALL
# merging until IDF is trustworthy) -- zero cross-category sharing, confirmed, at the cost of zero
# merging in that same window; bounded to a one-time bootstrap tax in production (4.2MB corpus
# clears n_docs>=20 almost immediately). NOT YET WIRED IN: open question whether the SAME n=1
# variance-floor leniency (mean_var above) re-creates this failure for any freshly-spawned brick
# inside an already-mature cluster (cluster n_docs>=20, new brick's own n=1) -- the floor formula
# only looks at the brick's own n, not n_docs, so this is untested, not ruled out. See
# lucy_west_cross_category_fix_candidates_tested_20261101.md for the full real numbers.
# ⛔ 2026-11-01, NORTH STAR CORRECTION, same day: the premise above ("two unlike bricks must not
# share") was WRONG. This router organizes by MATHEMATICAL signature similarity, never by subject
# matter -- route() doesn't need to know a line is about geography vs. biography, and two
# sufficiently similar signatures sharing a brick is INTENTIONAL, not a defect, even when a human
# would call their topics unrelated. Re-examined under the real evaluation axis (learning
# stability/interference, never topic purity): the shared brick's post-teach NLL (7.18 nats) was
# actually BETTER than an isolated, untouched brick's (7.89 nats) -- the data never showed sharing
# causing worse outcomes than not sharing. The force-spawn-during-cold-start fix above is very
# likely NOT the right thing to ship now -- it would trade away real mathematical self-organization
# to chase a topic-purity goal that was never the actual objective. The underlying engineering
# findings (unreliable small-sample IDF, permissive n=1 variance floor, Gaussian calibration)
# remain real and worth fixing -- any fix must preserve mathematical organization, evaluated by
# stability/retention, not by whether a human agrees with the apparent grouping. See
# lucy_west_mathematical_memory_ecology_north_star_20261101.md -- read before touching route() again.
# 🎯 2026-11-01, RESOLVED: tested whether the n=1 leniency above persists past n_docs>=20 -- it does
# NOT. At n_docs=30 (IDF genuinely active), a fresh n=1 singleton's margin test correctly separated
# 5/5 real test lines by real IDF-weighted cosine distance (close: cosine 0.85/0.86 -> margin
# +0.038/+0.042, accepted; far: cosine 0.04-0.14 -> margin -0.043 to -0.055, rejected), confirmed
# against both a hand-picked singleton and an independently-arising one. The leniency bug is
# COLD-START-SPECIFIC (IDF-inactive, <20 docs), not a standing flaw in this variance floor itself --
# see lucy_west_singleton_leniency_coldstart_only_confirmed_20261101.md for the full real numbers.

# 🔧 2026-10-09, Jonathan: "tighten merging for efficiency" -- measured result (30-pass, 500KB
# corpus): growth+consolidate SETTLES cleanly (105-106 bricks, stable from round 5 through round 30,
# not runaway) with real quality gains (held-out loss ~4.2-4.5, better than both the single net's
# 10.3-11.7 and the oracle-clustered bricks' own 6.0-6.6, both TRAIN loss, not a perfectly
# apples-to-apples comparison but a strong signal), BUT settles at ~1.05M total params -- still
# above the single D=384 net's 627K, the thing this architecture exists to beat on efficiency.
# Two real levers, not a stricter accept test (that would shrink merges, the wrong direction for
# "more efficient"): (1) consolidate() previously did ONE static ordered pass -- a merge early in
# the pass can make a LATER merge newly possible (the survivor's Phi_pos widens as it absorbs), but
# the single pass never revisited anything once scanned. (2) the merge accept test reused the exact
# routing-gate threshold (margin > 0) verbatim -- fine for ROUTING (an individual line should only
# join a brick it genuinely resembles), too strict for MERGING two already-existing, already-proven
# bricks, where some tolerance for being close-but-not-quite is reasonable since both sides already
# have real trained content behind them, not just one fresh line.
#
# ✅ REAL SWEEP, 2026-10-09, Jonathan: "find out were the safe line realy is... push it get real
# data ot a feeling" -- 7-point sweep (slack 1,2,3,4,5,6,8,12), same 30-pass/500KB methodology,
# each an independent run. The line is a CLIFF, not a slope:
#   slack 1-4: stable, low-churn settling, 92/88/70/63 bricks (949K/896K/722K/650K params), all
#              reasonable quality (held-out 4.1-5.5).
#   slack 5:   BORDERLINE UNSTABLE -- never truly settles, 50-60 bricks spawn+merge EVERY pass for
#              all 30 passes, bottomed at just 3 bricks on round 1, landed at 14 (144K params) only
#              because this run happened not to tip over.
#   slack 6+:  TOTAL COLLAPSE -- everything merges into a single brick (quality craters, train CE
#              5.4->7.6+). Confirmed at 6, 8, AND 12 -- not a fluke of one value.
# slack=4 (closest to the 627K single-net param target) sits ONE STEP from that cliff -- thin
# margin, not a safe operating point by itself. slack=3 has real margin below the instability zone
# AND produced the BEST held-out loss of the entire sweep (4.12, beating 1/2/4/5) -- not a
# safety/quality tradeoff, the safe choice also happens to be the best one measured. Honest cost:
# 722K params, still above the 627K target -- crossing that line appears to require the unstable
# zone, which this data shows risks catastrophic collapse, not a graceful tradeoff. Not chasing it.
#
# ⛔ SUPERSEDED, 2026-10-09 -- the above sweep was calibrated against FEATURE_DIM=64 with a SUMMED
# log_density. Real-mind collapse (1 brick from 59.7M chars) forced a 5-bug diagnostic arc
# (brick_cluster_production_port.md memory has the full chain): Phi_neg freeze + variance ceiling
# + candidate-K limiting (first three fixes, each real but each confirmed via direct
# instrumentation to NOT be the dominant mechanism) -> TF-IDF (fixed scale-dependent collapse but
# overcorrected diversity) -> THE REAL ROOT CAUSE: FEATURE_DIM=64 so undersized that real ~3000-
# trigram lines averaged ~46 hash collisions PER BUCKET, saturating all 64 buckets for every line
# regardless of content (confirmed: unrelated paragraphs on Cato the Younger, Sylvia Plath,
# SolarWinds, and Galatea-identity declarations all cascaded into one hub with +168 to +256
# log-density margins -- nowhere near borderline, so no slack value could have fixed it).
# FEATURE_DIM raised 64->8192 (see its own module comment) fixed the collision saturation but, as
# a direct side effect, broke THIS constant's calibration a second time -- log_density SUMMED
# over dims, so summing 128x more terms amplified any small systematic bias 128x, making
# slack=3.0 accept nearly everything (171 bricks collapsed toward 1 again). Fixed log_density to
# use np.mean instead of np.sum (see its own comment) -- margin scale is now independent of
# FEATURE_DIM. Re-swept MERGE_SLACK FRESH at this new scale (0.0-2.0 coarse, 0.12-0.28 fine,
# 30-pass stability check at 0.18/0.22/0.25): a smooth CONTINUOUS curve this time, not a sharp
# cliff -- 0.0->171, 0.1->153, 0.15->121, 0.18->99, 0.22->71, 0.25->55, 0.28->32, 0.5->8,
# 0.75->2, 1.0->1 (collapse). All of 0.18/0.22/0.25 held STABLE across the full 30 passes (zero
# churn after initial settling -- the real safety property, not just a snapshot). slack=0.22
# chosen: 71 bricks / 732,720 params lands almost exactly on the PRE-BUG healthy target (70
# bricks / 722,400 params) -- strong independent confirmation that the representation, not the
# threshold, was always the real lever.
# 🧛 2026-10-11, LUCY-SPECIFIC OVERRIDE (this file is F:\my-agent-vampire-lucy's own standalone
# copy, NOT D:\flybrain's production file -- this change does not touch the real Galatea mind).
# The 0.22 sweep above was tuned for the main production corpus: long, continuous natural-
# language prose where nearby lines are OFTEN genuinely topically related and SHOULD cluster.
# Lucy's vampire-teaching content is structurally different: each taught line is a short, atomic,
# usually UNRELATED Q&A fact ("User: what is the capital of France?\nLucy: Paris."), and real
# accepted-merge evidence (_diag_accepted_merges-style direct inspection of a 1000-fact run) showed
# bricks absorbing dozens of genuinely unrelated facts (geography+chemistry+literature all in one
# brick), consistent with 0.22 being too loose for this domain's much higher fraction of merely-
# superficially-similar ("What is X, innit?" template) pairs. Final host-off retention on that
# 1000-fact run was only 473/1000 (47.3%), down hard from 100% at 30-fact scale -- real, measured
# over-merging, not assumed. Tightening to 0.08 (~1/3 of 0.22) is a reasoned hypothesis, not a
# guess: NOT independently re-swept for this domain yet (that would need the same real sweep
# discipline D:\flybrain's own MERGE_SLACK calibration used) -- the next real 1000-fact run IS
# that test; compare its final retention against this run's 47.3% baseline before trusting 0.08
# as correct, and re-sweep properly if time allows rather than keep hand-picking.
MERGE_SLACK = 0.08   # mean-log-density units (see log_density's np.mean normalization); margin >
# -MERGE_SLACK accepted. This value is NOT comparable to the old sum-based sweep above -- re-
# verify against a real sweep, not a guess, if FEATURE_DIM or the log_density aggregation ever
# changes again (both silently rescale what this constant means, confirmed twice this session).
MERGE_MAX_ROUNDS = 10   # safety cap on the fixed-point iteration -- merges strictly reduce brick
# count each round so termination is guaranteed regardless, this just bounds worst-case cost.

MERGE_CANDIDATE_K = 10   # bound the number of bigger-brick candidates tested per merge decision --
# see the real multiple-comparisons bug this fixes in consolidate()'s own comment. Same role as
# WOLFPACK_GATE_K_NEIGHBORS in sovereign_core.py's own LSL-gate port.

ROUTE_REGION_INDEX_MIN_BRICKS = 400   # 2026-10-29: two-level (coarse-region + exact-within-region)
# routing fast path, validated in routing_scale_toy_test.py against this module's REAL
# _OnlineGaussian.log_density math, not a reimplementation. The toy sweep found NO real speedup
# below ~1000 bricks (region bookkeeping overhead eats the saving) and real brick counts today are
# ~145 (see brick_cluster_production_port.md) -- far under this floor, so route() stays on the
# original flat O(n_bricks) scan, bit-for-bit, until the cluster actually grows this large. Not a
# guess: picked with real headroom under the toy test's proven n=1000 operating point.
ROUTE_REGION_TOP_M = 6   # 2026-10-29 CORRECTED after the first choice (4) failed a harder check --
# the original toy harness (routing_scale_toy_test.py, synthetic bricks, n=1000, 60 queries) showed
# 0/60 mismatches at top_m=4, but that was a small sample, not proof of zero error -- flagged as a
# caveat before this was ever wired in. Re-tested directly against THIS module's real BrickCluster
# (real _OnlineGaussian fits, real route()/_route_fast()/_rebuild_regions(), n=500, 3 seeds x 150
# queries = 450 total): top_m=4 produced 2/450 REAL wrong-brick picks (near-tied margins, correct
# brick's region landing 5th-nearest by centroid distance -- just outside top_m=4). top_m=6 was the
# first setting with 0/450 mismatches across all 3 seeds; speedup at n=500 is a more modest ~2.45x
# (204 of 500 bricks scanned), not the originally-hoped 3.4x -- the honest number, not the
# optimistic one. See lucy_west_routing_fast_path_production_wired_20261029.md for the full account.
ROUTE_REGION_N_REGIONS_FLOOR = 2   # sqrt(n_bricks) regions, same as the toy test; floored so a
# degenerate tiny n_bricks never asks for 0 or 1 regions.

# Confirmation-gated vote mechanism (re-implemented 2026-11-07, Claude+Jonathan, from the real
# documented design in GALATEA_ENGINEER_DIARY_20261103_LUCYWEST_CONFIRMATION_GATED_VOTE_MECHANISM
# .txt + lucy_west_memory_architecture_locked_20261102.md). The original private patch to this
# file (wired 2026-11-02/03, real-tested end to end through this exact class) was lost when the
# USB drive it lived on failed on 2026-11-07. This is a faithful rebuild from the documented spec
# and the observed output schema of the lost teaching pipeline, not a guess -- same constants the
# diary records as tunable, not structural.
VOTE_MAX_VOTES = 10
VOTE_DECAY_BAND = 2       # max erosion _effective_vote ever subtracts; floor = MAX_VOTES-DECAY_BAND
VOTE_DECAY_AFTER = 15     # global_step ticks of "quiet" (no reconfirmation) per 1 point of erosion

# USAGE-BOOST, 2026-11-08 (Jonathan, recovered from memory this session -- see
# lucy_west_memory_architecture_locked_20261102.md's "USAGE-DRIVEN TEMPORARY CONFIRMATION-WEIGHT
# BOOST" section for the full worked rationale/examples). A SEPARATE, smaller, faster-decaying
# number from vote_count -- mere repetition/assertion (no independent check) gets a small,
# temporary nudge, never a permanent gain, so the illusory-truth-effect/propaganda exploit the
# confirmation gate already closes for vote_count can't reopen through a side door. Shaped like a
# real hormonal response per his own analogy ("adrenaline... excitement, fear"): a usage event
# SPIKES this straight to its cap, it does not ramp up by small increments; it then wears off
# (decays to ZERO, not a floor -- unlike vote_count's own decay, which never fully disappears)
# faster than vote_count's own decay clock, since it was never earned the way real confirmation is.
USAGE_BOOST_CAP = 2            # "not anything overpowering" -- same magnitude as VOTE_DECAY_BAND,
# deliberately small relative to VOTE_MAX_VOTES=10.
USAGE_BOOST_DECAY_AFTER = 5    # faster than VOTE_DECAY_AFTER=15 -- "short term yes, long term no."


class BrickCluster:
    """Growable cluster of FixedStepBrick, routed by an online LSL-style gate. V/D_brick fixed at
    construction (every brick the same size); the NUMBER of bricks grows as genuinely new content
    arrives that no existing brick's Phi_pos explains better than the shared Phi_neg."""
    def __init__(self, V, D_brick, seed):
        self.V = V
        self.D_brick = D_brick
        self.seed = seed
        self.bricks = []
        self.brick_pos = []       # one _OnlineGaussian per brick (Phi_pos) -- unbounded on purpose,
                                   # should reflect everything that brick has actually absorbed.
        self.pending_visited = []  # parallel to self.bricks: pending_visited[i] is a list of
                                   # frozensets, one per packet currently queued in
                                   # bricks[i].pending_relay (same order) -- the growing set of
                                   # brick-indices that specific packet has already passed through.
                                   # Owned here, not on the brick, because only the cluster has
                                   # sibling visibility; see _relay_excess for the loop-proof rule
                                   # this enforces (2026-10-28 redesign).
        self.neg = _OnlineGaussian(freeze_at_n=NEG_FREEZE_AT_N)   # shared Phi_neg, BOUNDED --
                                   # see _OnlineGaussian's freeze_at_n docstring for why.
        # TF-IDF document-frequency tracking (the real root-cause fix, see IDF_MIN_DOCS/
        # IDF_SMOOTHING's module comment) -- doc_freq[k] = how many LINES contained trigram-hash
        # bucket k at all (presence, not raw count); n_docs = total lines seen. Unbounded on
        # purpose, unlike Phi_neg -- IDF weights are meant to reflect the TRUE corpus-wide
        # commonness of each bucket and should keep refining as more real data is seen, not freeze.
        self.doc_freq = np.zeros(FEATURE_DIM)
        self.n_docs = 0
        self._gvar_cache = None   # see _global_var -- same cache-by-source-stat-change idea as
        self._gvar_cache_neg_n = -1   # _OnlineGaussian's, keyed on neg.n (changes IFF neg's own
        # stats change). Once neg freezes at NEG_FREEZE_AT_N, this returns the SAME array object
        # every call, which also lets every brick's own _cached_stats (keyed partly on
        # id(global_var)) hit its cache instead of recomputing per line.
        self._region_centroids = None   # route()'s two-level fast-path index (see
        self._region_members = None     # ROUTE_REGION_INDEX_MIN_BRICKS's comment). Only ever
        self._region_fresh = False      # built by _rebuild_regions(), called from consolidate()
        # against the just-settled brick population; _spawn() marks it stale again immediately
        # (a freshly spawned brick belongs to no region yet) so route() can never silently use an
        # index that's missing bricks that actually exist -- when stale OR under the min-bricks
        # floor, route() falls back to the original exact flat scan, never an approximation.
        self.brick_taught_lines = []   # parallel to self.bricks: brick_taught_lines[i] is the list
        # of verbatim line_text strings ever routed to and trained into brick i. 2026-10-29,
        # Jonathan's "question the premise" finding: a single full-BPTT gradient step on a fresh
        # 40-dim brick measured WORSE than random (loss 10.4 vs ln(256)=5.54) -- a freshly-taught
        # one-shot fact brick cannot be expected to GENERATE a fluent answer from its own weights,
        # no matter how it's routed or ensembled (tested directly, see
        # scattered_fact_ensemble_test.py / lucy_west_retrieval_not_generation_20261029.md). The
        # fix is to stop asking it to generate at all for this case: retrieve the exact taught
        # text instead (it already contains the donor's real answer verbatim, see learn_line's
        # caller in brickcluster_engine.py -- "User: ...\nLucy: ..."), and only fall back to
        # cluster.generate() when no stored line exists for the matched brick. Already-trained-at-
        # checkpoint-time bricks (before this field existed) simply have an empty list here and
        # correctly fall back to the old generate() path, not a bug, see field's own doc above.
        self.vote_count = []         # parallel to self.bricks -- confirmation-gated vote, see
        self.last_confirm_step = []  # VOTE_MAX_VOTES's module comment for provenance. vote_count
        self.ceiling_reached = []    # starts at 1; last_confirm_step/ceiling_reached drive decay.
        self._global_step = 0        # increments on EVERY gated learn_line() call, cluster-wide
        # (not per-brick) -- a wall-clock-like proxy for "time/activity has passed," deliberately
        # NOT a per-brick usage counter (that would reopen the usage/confirmation conflation this
        # mechanism exists to close).
        self.last_usage_step = []    # parallel to self.bricks -- USAGE_BOOST_CAP's own module
        # comment for provenance. -1 = never had a usage event (so _effective_usage_boost reads 0).
        self.challenger_line = []    # parallel to self.bricks -- the new, contradicting line_text
        # currently accumulating its OWN proof against this brick's incumbent belief, or "" if none
        # pending. See learn_line's CONTRADICTION branch below.
        self.challenger_vote = []    # parallel to self.bricks -- how many real contradiction events
        # have offered THIS SAME challenger_line so far. A genuinely different new challenger text
        # replaces (does not add to) whatever was pending -- two different alternatives can't share
        # one tally.

    def tfidf_sparse(self, line):
        """The real hot-path feature computation -- returns (idx, val), NEVER materializes a
        dense FEATURE_DIM array (see raw_trigram_sparse's comment for why that matters at
        FEATURE_DIM=8192). Same TF-IDF math as the now-legacy tfidf_features below, just sparse."""
        counts = raw_trigram_sparse(line, FEATURE_DIM)
        if not counts:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
        cnt = np.fromiter(counts.values(), dtype=np.float64, count=len(counts))
        if self.n_docs < IDF_MIN_DOCS:
            val = cnt
        else:
            df = self.doc_freq[idx]
            idf = np.log((self.n_docs + IDF_SMOOTHING) / (df + IDF_SMOOTHING))
            val = cnt * idf
        norm = np.linalg.norm(val)
        if norm > 0:
            val = val / norm
        return idx, val

    def tfidf_features(self, line):
        """Dense legacy accessor -- kept ONLY for existing diagnostics that expect a dense array
        (_diag_merge_mechanics.py etc.); the real pipeline (route/learn_line/forward_logits/
        generate) uses tfidf_sparse instead, see its docstring for why. Same exact values, just
        scattered into a dense vector afterward."""
        idx, val = self.tfidf_sparse(line)
        v = np.zeros(FEATURE_DIM)
        v[idx] = val
        return v

    def _update_doc_freq(self, line):
        counts = raw_trigram_sparse(line, FEATURE_DIM)
        if counts:
            idx = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
            self.doc_freq[idx] += 1.0
        self.n_docs += 1

    def _global_var(self):
        if self.neg.n < 2:
            return np.ones(FEATURE_DIM)
        if self.neg.n != self._gvar_cache_neg_n:
            mean = self.neg.sum / self.neg.n
            self._gvar_cache = np.maximum(self.neg.sumsq / self.neg.n - mean * mean, 1e-6)
            self._gvar_cache_neg_n = self.neg.n
        return self._gvar_cache

    def _spawn(self):
        idx = len(self.bricks)
        self.bricks.append(FixedStepBrick(self.V, self.D_brick, seed=self.seed + 1000 * idx))
        self.brick_pos.append(_OnlineGaussian())
        self.pending_visited.append([])
        self.brick_taught_lines.append([])
        self.vote_count.append(1)
        self.last_confirm_step.append(self._global_step)
        self.ceiling_reached.append(False)
        self.last_usage_step.append(-1)
        self.challenger_line.append("")
        self.challenger_vote.append(0)
        self._region_fresh = False   # the new brick belongs to no region in the old index yet --
        # see ROUTE_REGION_INDEX_MIN_BRICKS's comment; route() falls back to the exact flat scan
        # until the next consolidate() call rebuilds the index over the now-current population.
        return idx

    def route(self, features, for_training):
        """Returns a brick index. TRAINING may spawn a new brick if nothing existing fits;
        INFERENCE never spawns -- always returns the best-available brick (spawning on a
        read-only call would mean generating from an untrained net, which is never useful),
        matching LSL's own training/inference asymmetry.

        🔁⛔ 2026-10-09, PERSISTENT CHURN at 2M+ char scale -- DIAGNOSED (confirmed, not guessed)
        but the obvious fix was TRIED AND REVERTED, a real negative result worth keeping.
        Diagnosed directly via _diag_churn_mechanism.py: at larger corpora, ~280 bricks were
        spawning fresh EVERY SINGLE PASS, immediately merged back down by that same pass's
        consolidate() call, forever (confirmed: 272/280 of the exact same lines spawned in 4/4
        watched passes, 97-100% overlap pass-over-pass -- a deterministic cycle, not noise). Root
        cause: routing required `margin > 0` (strict) while consolidate() immediately tested the
        SAME underlying margin at `margin > -MERGE_SLACK` (looser). Any line whose best existing-
        brick margin fell in (-MERGE_SLACK, 0] failed routing (spawned fresh) then passed merging
        in the very same pass (got swallowed) -- every pass, forever.
        TRIED: changing this threshold to `margin <= -MERGE_SLACK` (reusing consolidate()'s own
        tolerance, pre-empting a merge that would happen moments later anyway). CONFIRMED to
        eliminate the 2M-scale churn completely (0 merges across 4 passes, down from ~280/pass,
        re-verified via the same diagnostic). BUT regression-tested against the already-shipped,
        verified-healthy 500K scale and found a REAL regression: brick count collapsed 71->4.
        The two decisions are NOT equivalent at smaller scale the way the 2M diagnosis suggested --
        route() picks the single best-by-margin brick over ALL existing bricks, while
        consolidate() only considers the K=10 NEAREST-BY-DISTANCE candidates and processes merges
        in a specific size-ordered, cascading sequence within a round; loosening ROUTING's
        threshold globally let many more lines merge into existing bricks during growth itself
        (not just the narrow churn-causing band), collapsing diversity broadly, not narrowly.
        REVERTED to the original strict `margin > 0`. The 2M+ churn remains open and unfixed --
        confirmed non-dangerous (stable brick count/quality, see brick_cluster_production_port.md
        memory), just wasteful. A real fix would need something more targeted than reusing
        MERGE_SLACK wholesale -- e.g. a per-line "already resolved this pass" memory, or aligning
        consolidate()'s candidate selection with route()'s argmax-over-all-bricks instead of
        K-nearest-by-distance -- deferred, not attempted again without a plan for why THAT
        wouldn't also need a fresh regression check at every scale, per this exact negative result."""
        if not self.bricks:
            return self._spawn() if for_training else None
        gvar = self._global_var()
        # 2026-10-29 FIX, real bug confirmed live: neg_ld used to default to -np.inf whenever
        # self.neg.n < 2 (the shared background model hasn't seen 2+ examples system-wide yet),
        # which makes margin = pos_ld - (-inf) = +inf UNCONDITIONALLY for every brick, regardless
        # of actual content fit -- route() would then NEVER spawn (margin can't be <=0) and would
        # force literally every early query into whichever brick already exists, no matter how
        # unrelated. Confirmed as a real contributor to catastrophic brick contamination (a real
        # brick in the live checkpoint absorbed hundreds of mismatched queries this way, teacher-
        # forced CE on its own taught fact measured 12.47 nats, worse than random 5.55 -- see
        # lucy_west_single_brick_generation_root_cause_20261029.md). Fixed by using a NEUTRAL 0.0
        # sentinel instead of -inf: margin degrades to the brick's raw (uncalibrated) log_density,
        # which is still a real, usable signal even before the background model stabilizes --
        # confirmed empirically (routing_coldstart_margin_fix_test.py) that a freshly-spawned
        # brick's raw pos_ld on both a similar AND an unrelated query comes out identical and
        # NEGATIVE at n=1 (the Gaussian genuinely can't discriminate yet with one example), so this
        # correctly spawns separate bricks for both rather than force-merging either -- the safe,
        # conservative default, not a guess. Only touches the neg.n<2 window (the first ~1-2 real
        # facts ever taught to a cluster); has no effect once the background model has real data.
        neg_ld = self.neg.log_density(features, VAR_FLOOR_FRAC, gvar) if self.neg.n >= 2 else 0.0
        if self._region_fresh and len(self.bricks) >= ROUTE_REGION_INDEX_MIN_BRICKS:
            best_idx, best_margin = self._route_fast(features, gvar, neg_ld)
        else:
            margins = [pos.log_density(features, VAR_FLOOR_FRAC, gvar) - neg_ld for pos in self.brick_pos]
            best_idx = int(np.argmax(margins))
            best_margin = margins[best_idx]
        if for_training and best_margin <= 0.0:
            return self._spawn()
        return best_idx

    def _route_fast(self, features, gvar, neg_ld):
        """Two-level routing: nearest ROUTE_REGION_TOP_M region centroids (cheap), then the SAME
        exact log_density margin computation as the flat scan, restricted to just those regions'
        bricks. Validated in routing_scale_toy_test.py: 0/60 mismatches against the flat scan at
        top_m=ROUTE_REGION_TOP_M, n_bricks=1000. Only ever called when self._region_fresh is True
        (see route()), i.e. the index exactly matches the current self.bricks population."""
        dense = np.zeros(FEATURE_DIM)
        idx, val = features
        dense[idx] = val
        d = np.linalg.norm(self._region_centroids - dense[None, :], axis=1)
        chosen_regions = np.argsort(d)[:ROUTE_REGION_TOP_M]
        best_idx, best_margin = -1, -np.inf
        for r in chosen_regions:
            for i in self._region_members[r]:
                m = self.brick_pos[i].log_density(features, VAR_FLOOR_FRAC, gvar) - neg_ld
                if m > best_margin:
                    best_margin, best_idx = m, i
        return best_idx, best_margin

    def _rebuild_regions(self):
        """Rebuilds route()'s two-level region index (k-means over each brick's real Phi_pos mean,
        same recipe as routing_scale_toy_test.py). Called only from consolidate() -- the existing
        "sleep" re-clustering cycle -- so the index is always built against the population
        consolidate() just finished settling, never a half-updated one. _spawn() marks the index
        stale the instant a new brick appears; it only becomes fresh again at the next
        consolidate() call, so route()'s fast path can never see a brick population the index
        doesn't actually match."""
        n = len(self.bricks)
        if n < ROUTE_REGION_INDEX_MIN_BRICKS:
            self._region_fresh = False
            return
        gvar = self._global_var()
        means = np.stack([pos.mean_var(VAR_FLOOR_FRAC, gvar)[0] for pos in self.brick_pos])
        n_regions = max(ROUTE_REGION_N_REGIONS_FLOOR, int(np.sqrt(n)))
        rng = np.random.default_rng(self.seed)
        idx0 = rng.choice(n, size=min(n_regions, n), replace=False)
        centroids = means[idx0].copy()
        assign = np.zeros(n, dtype=np.int64)
        for _ in range(8):
            d = np.linalg.norm(means[:, None, :] - centroids[None, :, :], axis=2)
            assign = np.argmin(d, axis=1)
            for r in range(centroids.shape[0]):
                members = means[assign == r]
                if len(members):
                    centroids[r] = members.mean(axis=0)
        region_members = [[] for _ in range(centroids.shape[0])]
        for i, r in enumerate(assign):
            region_members[r].append(i)
        self._region_centroids = centroids
        self._region_members = region_members
        self._region_fresh = True

    def _effective_vote(self, idx):
        """Confirmation-gated vote mechanism, decay-to-floor piece. Deliberately a READ view,
        never mutates stored vote_count -- only real contradiction events (in learn_line below)
        physically decrement storage. A fresh reconfirmation on a fully-decayed trace therefore
        snaps immediately back to the ceiling in one call, not a slow climb -- confirming now
        legitimately proves both "still true" and "just rechecked," full credit for both at once.
        The clock (_global_step) is cluster-wide, not per-brick -- see its own field comment."""
        if not self.ceiling_reached[idx]:
            return self.vote_count[idx]   # never decays before it's ever been maxed
        quiet = self._global_step - self.last_confirm_step[idx]
        erosion = min(VOTE_DECAY_BAND, quiet // VOTE_DECAY_AFTER)
        return max(0, self.vote_count[idx] - erosion)

    def _effective_usage_boost(self, idx):
        """USAGE_BOOST_CAP's own module comment for provenance. A READ view, same spirit as
        _effective_vote -- never mutates storage. Decays fully to 0 (not a floor): the boost was
        never earned the way real confirmation is, so unlike vote_count it has nothing to protect
        once it's worn off. -1 (never used) reads as 0, same as fully decayed."""
        if self.last_usage_step[idx] < 0:
            return 0
        quiet = self._global_step - self.last_usage_step[idx]
        erosion = min(USAGE_BOOST_CAP, quiet // USAGE_BOOST_DECAY_AFTER)
        return max(0, USAGE_BOOST_CAP - erosion)

    def learn_line(self, batches_for_line, line_text, lr_scale, gated=False, confirmed=None):
        """batches_for_line: the windowed batches (same bucket-by-length convention as
        lutgtlm_ingest.py) for ONE line. Routes (possibly spawning), trains, updates the gate.
        Takes the raw line TEXT (not pre-computed features, unlike the original version) so TF-IDF
        weighting can use the CURRENT doc-frequency stats before updating them with this line --
        computing features externally would make doc-frequency tracking inconsistent/bypassable.

        gated=False (default): legacy/ungated, byte-identical to the old behavior -- every real
        production call site that doesn't pass these new params is provably unaffected.
        gated=True makes `confirmed` a real three-way signal, closing the pure-repetition illusory-
        truth exploit (a bare belief could flip in as few as 8 unverified reps):
          confirmed=None  -> USAGE. Never trains, never touches vote_count. 2026-11-08: now spikes
                              last_usage_step (see USAGE_BOOST_CAP/_effective_usage_boost) -- a
                              small, separate, fully-fading nudge, never a vote_count change.
          confirmed=True  -> CONFIRMATION. vote_count += 1 (capped at VOTE_MAX_VOTES), decay clock
                              resets, then trains normally (reinforces/extends current content).
                              2026-11-08: also clears any pending challenger (see below) -- a real
                              reconfirmation of the incumbent invalidates an in-progress rival.
          confirmed=False -> CONTRADICTION. 2026-11-08 rebuild (Jonathan: "she has to compare truth
                              before she decides which one to keep" -- a new challenger must prove
                              its OWN confirmation weight, not just win by outlasting the
                              incumbent's erosion). vote_count -= 1 as before (the incumbent's own
                              confidence still erodes under contradiction). SEPARATELY, this
                              call's line_text accumulates its own challenger_vote (reset to 1 if
                              it's a genuinely new challenger; +1 if the SAME challenger has been
                              offered before). The brick only flips once challenger_vote actually
                              reaches/exceeds the incumbent's current effective_vote -- a real
                              comparison, not attrition against a fixed floor. On flip: old taught
                              content discarded, the challenger's own earned vote becomes the new
                              incumbent's vote_count (not reset to 1 -- proven weight carries over),
                              and THIS call's line is the one that actually trains.
        Returns (idx, mean_loss_or_None) either way -- None for a blocked/no-op call, exactly
        like the legacy path returns None when batches_for_line is empty."""
        features = self.tfidf_sparse(line_text)
        idx = self.route(features, for_training=True)
        if not gated:
            return self._train_brick(idx, features, batches_for_line, line_text, lr_scale)
        self._global_step += 1
        if confirmed is None:
            self.last_usage_step[idx] = self._global_step
            return idx, None
        if confirmed:
            self.vote_count[idx] = min(VOTE_MAX_VOTES, self.vote_count[idx] + 1)
            self.last_confirm_step[idx] = self._global_step
            if self.vote_count[idx] >= VOTE_MAX_VOTES:
                self.ceiling_reached[idx] = True
            self.challenger_line[idx] = ""
            self.challenger_vote[idx] = 0
            return self._train_brick(idx, features, batches_for_line, line_text, lr_scale)
        # confirmed is False -- CONTRADICTION
        self.vote_count[idx] = max(0, self.vote_count[idx] - 1)
        if self.challenger_line[idx] == line_text:
            self.challenger_vote[idx] += 1
        else:
            self.challenger_line[idx] = line_text
            self.challenger_vote[idx] = 1
        if self.challenger_vote[idx] < self._effective_vote(idx):
            return idx, None   # stale belief still holds more weight than the challenger -- blocked
        # challenger's own proven weight now matches or exceeds the incumbent's -- flip: discard
        # old taught content, the challenger's earned vote becomes the new incumbent's vote_count.
        self.brick_taught_lines[idx] = []
        self.vote_count[idx] = self.challenger_vote[idx]
        self.ceiling_reached[idx] = self.vote_count[idx] >= VOTE_MAX_VOTES
        self.last_confirm_step[idx] = self._global_step
        self.challenger_line[idx] = ""
        self.challenger_vote[idx] = 0
        return self._train_brick(idx, features, batches_for_line, line_text, lr_scale)

    # 2026-11-08: the open "generation-quality bug" (_lucy_candidate's own docstring: "cluster.
    # generate()'s own single-shot net output -- STILL the pre-existing, still-open generation-
    # quality bug, can be garbled"). Root cause confirmed LIVE against the real checkpoint, not
    # guessed: production only ever gives a brick ONE gradient-step exposure per taught fact
    # (this loop ran once, over batches_for_line, every call). The 2026-11-01 validated-subset
    # breakthrough (lucy_west_validated_subset_exact_generation_breakthrough_20261101.md) that IS
    # wired in here (VALIDATED_TOP_N=None, uncapped, above) only ever reached exact correct
    # generation after 8-30+ REPEATED learn_batch calls on the SAME fact -- never claimed a single
    # exposure would converge. Tested directly against 180 real single-exposure production bricks
    # from the live 223-item diary run: 0/180 reproduced their own taught answer on generate(),
    # all garbled -- confirming this is exactly the rep-count gap, not a regression or a different
    # bug. Fix: repeat the SAME batches_for_line call multiple times for a brick's FIRST exposure
    # to a genuinely new fact, matching the breakthrough's own measured convergence window.
    #
    # Bounded by answer length, NOT computed precisely from it -- Jonathan's own stated principle
    # ("humans sort their tools not by measurement but category and frequency of use") turned out
    # to be the right engineering call here too, not just a nice metaphor: live-measured cost on a
    # real 849-character taught answer was ~100-135 SECONDS per rep (the uncapped validated-subset
    # search re-evaluates every candidate flip with a full forward pass; cost scales with sequence
    # length). 20 reps on an answer that long would add ~35+ minutes to ONE teach call -- not
    # viable. Rather than compute a precise per-fact rep count from exact answer length (fragile,
    # and the cost curve isn't even characterized well enough yet to do that honestly), facts are
    # bucketed into one category ("short," below) that gets the full, already-proven convergence
    # treatment, and everything else is left on the existing single-pass + retrieval-verbatim path,
    # which is already safe and already shipped. This is a first calibration point, same spirit as
    # RETRIEVAL_COSINE_FLOOR's own comment -- not a tuned production boundary.
    NEW_FACT_CONVERGENCE_REPS = 20  # breakthrough table: uncapped budget reaches NLL~0.004-0.040
    # by rep8-20 on its one tested fact; 20 gives real margin without chasing diminishing returns
    # past that (rep40 only improved 0.004->0.002).
    NEW_FACT_REPS_MAX_ANSWER_CHARS = 80  # "short" category boundary. The ONE fact actually proven
    # to converge (2026-11-01 breakthrough) was 32 chars ("The capital of France is Paris."); 80
    # gives real headroom above that proven point while staying well clear of the 849-char case
    # that measured ~100-135s/rep live tonight -- not derived from a cost curve (only 2 real data
    # points exist), a deliberately conservative category line, not a precise threshold.

    def _train_brick(self, idx, features, batches_for_line, line_text, lr_scale):
        """The actual weight update + bookkeeping -- factored out of learn_line (2026-11-07, same
        patch that added gated/confirmed above) so the confirmation gate can decide block-vs-train
        BEFORE any of this runs, rather than after. Body is otherwise unchanged from the original
        single-function version, except for the new-fact convergence-reps loop (2026-11-08, see
        the constants' own comment above)."""
        brick = self.bricks[idx]
        # 2026-10-29: read reuse count BEFORE this call's own update (below) touches it -- the real
        # isolation finding behind BIAS_CORROB_REUSE_THRESHOLD needs "how many times was this brick
        # reactivated BEFORE now," not including the activation happening this call.
        corrob_eligible = self.brick_pos[idx].n >= BIAS_CORROB_REUSE_THRESHOLD
        old_visited = self.pending_visited[idx]
        self.pending_visited[idx] = []
        losses = []
        own_excess_b = np.zeros(brick.D, np.float64)
        own_excess_bg = np.zeros(brick.D, np.float64)
        relay_events = []  # (excess_b, excess_bg, visited) -- packets that arrived via relay and
                            # still didn't fit keep THEIR OWN carried-forward history; this brick's
                            # own fresh shedding (below) starts a brand-new history at this brick.
        is_new_fact = line_text not in self.brick_taught_lines[idx]
        answer_len = len(line_text.split("\nLucy:", 1)[1]) if "\nLucy:" in line_text else len(line_text)
        reps = (self.NEW_FACT_CONVERGENCE_REPS
                if is_new_fact and answer_len <= self.NEW_FACT_REPS_MAX_ANSWER_CHARS else 1)
        for rep in range(reps):
            for b in batches_for_line:
                losses.append(float(brick.learn_batch(b, lr_scale=lr_scale, corrob_eligible=corrob_eligible).mean()))
                own_excess_b += brick._last_excess_b
                own_excess_bg += brick._last_excess_bg
                for (exb, exbg), old_v in zip(brick._last_relay_excess, old_visited):
                    if np.any(exb) or np.any(exbg):
                        relay_events.append((exb, exbg, old_v | {idx}))
                old_visited = []  # only the FIRST learn_batch call across ALL reps can consume
                                   # anything (pending_relay is fully drained the instant it's
                                   # non-empty) -- reps 2+ never see pending relay packets at all.
        self.brick_pos[idx].update(features)
        self.neg.update(features)
        self._update_doc_freq(line_text)
        if line_text not in self.brick_taught_lines[idx]:   # dedupe -- the caller's own exact-
            self.brick_taught_lines[idx].append(line_text)   # duplicate fast path already skips
            # most repeats before learn_line ever runs, this is just a defensive second guard
            # against unbounded growth, not the primary dedupe mechanism.
        if np.any(own_excess_b) or np.any(own_excess_bg):
            relay_events.append((own_excess_b, own_excess_bg, frozenset({idx})))
        for exb, exbg, visited in relay_events:
            self._relay_excess(idx, exb, exbg, visited)
        return idx, (sum(losses) / len(losses) if losses else None)

    def _relay_excess(self, source_idx, excess_b, excess_bg, visited):
        """PairedRelay dam/cups mechanism, loop-proof BY CONSTRUCTION (2026-10-28 redesign, per
        Jonathan's explicit requirement: "at no time can the signal ever return... it has to go
        elsewhere"). `visited` is the set of every brick this specific packet has already passed
        through (including source_idx, added below before queuing); a relay NEVER targets anything
        in that set, so a chain can never revisit anywhere it has already been -- not just the
        immediate sender (which only stops simple back-and-forth 2-cycles) but the WHOLE path so
        far, which also stops longer N-cycles and the mutual-top-similarity-clique bounce confirmed
        directly (repair_bias_runaway.py's dry run against the real backup checkpoint: bricks 65
        and 128 -- each other's #1 most-similar sibling by content -- bounced ~150M back and forth
        in near-CONSTANT magnitude for 20000+ hops under the old source-only exclusion, never
        actually draining). Termination is now guaranteed structurally: at most len(bricks) hops
        total since the history only grows, and in practice the ceiling-sized chunk kept at every
        hop (BIAS_CEIL) means the shed amount roughly halves each time it's split across targets, so
        it disintegrates (produces zero further excess, naturally via _clip_and_shed) long before
        the history could ever be exhausted. Split across the (up to) two most SIMILAR eligible
        other bricks, ranked by the same Phi_pos content-similarity space already used for routing
        and merging. If fewer than 2 (or 0) eligible bricks remain, relays to however many are
        actually available, or drops it if none are left -- an honest terminal case ("nowhere left
        to send it"), not a silent assumption that candidates always exist. Also DISINTEGRATES
        (destroys the excess rather than relaying) once `visited` has grown past MAX_RELAY_HOPS --
        see that constant's comment for why an unbounded-depth cascade is a real combinatorial
        problem, not just a slow one."""
        if len(visited) > MAX_RELAY_HOPS:
            return
        candidates = [j for j in range(len(self.bricks)) if j not in visited and self.brick_pos[j].n > 0]
        if not candidates:
            return
        gvar = self._global_var()
        src_mean, _ = self.brick_pos[source_idx].mean_var(VAR_FLOOR_FRAC, gvar)
        sims = []
        for j in candidates:
            mean_j, _ = self.brick_pos[j].mean_var(VAR_FLOOR_FRAC, gvar)
            denom = (np.linalg.norm(src_mean) * np.linalg.norm(mean_j)) or 1.0
            sims.append(float(np.dot(src_mean, mean_j) / denom))
        order = np.argsort(sims)[::-1]
        targets = [candidates[i] for i in order[:2]]
        share = 1.0 / len(targets)
        new_visited = visited | {source_idx}
        for j in targets:
            self.bricks[j].pending_relay.append((excess_b * share, excess_bg * share))
            self.pending_visited[j].append(new_visited)

    def consolidate(self):
        """Periodic re-merge pass (2026-10-09), closer to what k-means does globally -- the online,
        one-line-at-a-time growth policy is greedy: a brick that claims a region early is never
        revisited even when a later, better-established brick would have been a fitter home, so
        growth alone over-spawns relative to the oracle k-means clustering this architecture was
        proven against (measured: 148 bricks for 224 lines, ~1.48M total params -- MORE than the
        single D=384 net's 627K this was built to beat, undermining the whole efficiency case).

        No weight-averaging -- two independently-initialized quinary nets' weight spaces are not
        aligned, averaging them would not produce something that behaves like either (unlike
        Wolfpack's prototype nudge, which operates in a shared, meaningful coordinate space).
        Instead: sort bricks by how much they've actually absorbed (pos.n, most-trained first) and
        for each smaller brick, ask whether a BIGGER brick's own Phi_pos already accepts the
        smaller one's mean -- the exact same accept/reject mechanism already proven for routing,
        reused for merging rather than invented fresh. One-sided (does the bigger brick accept the
        smaller one's territory), not mutual -- the smaller brick's own Phi_pos is often still
        floor-inflated (see _OnlineGaussian's adaptive floor) and an unreliable judge of whether IT
        would accept the bigger brick back; the bigger, more-established brick's judgment is the
        trustworthy one here.
        If accepted: the smaller brick's (n, sum, sumsq) sufficient statistics are EXACTLY absorbed
        into the survivor's Phi_pos (plain vector addition -- this is not an approximation, it is
        the same identity that makes online/parallel variance algorithms exact) and the smaller
        brick's quinary net is discarded. The content it had trained on is not replayed into the
        survivor -- an honest tradeoff, not hidden: these are typically lightly-trained bricks (1-3
        lines) to begin with, so little is actually lost, and future similar lines will now route to
        and keep training the survivor instead of re-spawning.

        Runs to a FIXED POINT (up to MERGE_MAX_ROUNDS single scans), not just one static pass -- a
        merge early in a scan widens the survivor's Phi_pos, which can make a DIFFERENT, previously-
        rejected small brick newly mergeable; a single pass never revisited those. Uses MERGE_SLACK
        (looser than the routing gate's own margin>0 threshold on purpose, see the module comment)
        since merging two already-trained bricks can tolerate more slack than routing one fresh
        line can. Returns the TOTAL number of merges made across every round."""
        self._region_fresh = False   # about to possibly reindex self.bricks/self.brick_pos
        # (merges below) -- any region index built before this point is unsafe to use mid-pass;
        # rebuilt fresh at the end against whatever population this call settles on.
        total_merged = 0
        for _round in range(MERGE_MAX_ROUNDS):
            if len(self.bricks) < 2:
                break
            order = sorted(range(len(self.bricks)), key=lambda i: self.brick_pos[i].n, reverse=True)
            gvar = self._global_var()
            neg_ld_cache = {}

            def neg_ld(x_key, x):
                # Same -inf-sentinel bug and same fix as route()'s own comment -- during
                # neg.n<2 cold start this used to make margin unconditionally > -MERGE_SLACK,
                # forcing every smaller brick to merge into the biggest one regardless of fit.
                if x_key not in neg_ld_cache:
                    neg_ld_cache[x_key] = self.neg.log_density(x, VAR_FLOOR_FRAC, gvar) if self.neg.n >= 2 else 0.0
                return neg_ld_cache[x_key]

            absorbed_into = {}   # smaller brick index -> surviving brick index
            for rank, small_i in enumerate(order):
                if small_i in absorbed_into:
                    continue
                small_pos = self.brick_pos[small_i]
                if small_pos.n == 0:
                    continue
                small_mean = small_pos.sum / small_pos.n
                # MERGE_CANDIDATE_K (2026-10-09, real bug found by directly instrumenting a merge
                # pass on the actual 1.5M-char corpus, not guessed): candidates used to be EVERY
                # bigger-or-equal-ranked brick -- order[:rank], which is up to hundreds at real
                # scale (444 bricks measured). Confirmed directly: individual pairwise margins were
                # correctly large and negative (-20 to -47 against the single biggest brick, well
                # below -MERGE_SLACK=-3, correctly rejected) -- the per-pair logic was never wrong.
                # But testing hundreds of independent candidates per decision makes it near-certain
                # SOME candidate clears the bar by chance, a real multiple-comparisons problem, not
                # a variance or background-scale problem (which is exactly why capping variance via
                # VAR_CEILING_MULT changed nothing -- that fix addressed a mechanism that was never
                # the active one). Bounding candidates to the K nearest by distance, same spirit as
                # Sovereign Core's own WOLFPACK_GATE_K_NEIGHBORS, caps the number of trials per
                # decision regardless of total population size.
                candidates = [big_i for big_i in order[:rank] if big_i not in absorbed_into and big_i != small_i]
                candidates.sort(key=lambda big_i: float(np.sum(
                    (small_mean - self.brick_pos[big_i].sum / max(self.brick_pos[big_i].n, 1)) ** 2)))
                for big_i in candidates[:MERGE_CANDIDATE_K]:
                    big_pos = self.brick_pos[big_i]
                    margin = big_pos.log_density(small_mean, VAR_FLOOR_FRAC, gvar) - neg_ld(small_i, small_mean)
                    if margin > -MERGE_SLACK:
                        big_pos.n += small_pos.n
                        big_pos.sum += small_pos.sum
                        big_pos.sumsq += small_pos.sumsq
                        for ln in self.brick_taught_lines[small_i]:   # survivor must keep
                            if ln not in self.brick_taught_lines[big_i]:   # retrieving BOTH
                                self.brick_taught_lines[big_i].append(ln)   # facts post-merge,
                        # not just the bigger brick's own -- see brick_taught_lines field doc.
                        absorbed_into[small_i] = big_i
                        break

            if not absorbed_into:
                break
            keep = [i for i in range(len(self.bricks)) if i not in absorbed_into]
            self.bricks = [self.bricks[i] for i in keep]
            self.brick_pos = [self.brick_pos[i] for i in keep]
            # Reindex pending_visited in lockstep (2026-10-28) -- without this, after a merge
            # shifts indices, pending_visited[i] would silently refer to the WRONG surviving
            # brick's queue, corrupting the loop-proof relay's own bookkeeping. Any pending relay
            # queued specifically on a brick that got merged AWAY is dropped along with the rest of
            # that brick's unique state -- the same already-accepted tradeoff as the merge itself,
            # not a new kind of loss.
            self.pending_visited = [self.pending_visited[i] for i in keep]
            self.brick_taught_lines = [self.brick_taught_lines[i] for i in keep]
            total_merged += len(absorbed_into)
        self._rebuild_regions()   # see its own docstring -- rebuilds route()'s fast-path index
        # against the now-settled population (inert, a no-op flag-set, below ROUTE_REGION_INDEX_MIN_BRICKS).
        return total_merged

    def forward_logits(self, idxs, line_text):
        features = self.tfidf_sparse(line_text)
        idx = self.route(features, for_training=False)
        if idx is None:
            return None
        return self.bricks[idx].forward_logits(idxs)

    def generate(self, seed_idx, itos, line_text, n=60, temp=0.7, rep_pen=1.0, seed=0):
        features = self.tfidf_sparse(line_text)
        idx = self.route(features, for_training=False)
        if idx is None:
            return ""
        return self.bricks[idx].generate(seed_idx, itos, n=n, temp=temp, rep_pen=rep_pen, seed=seed)

    def export_state(self):
        state = {"n_bricks": np.array(len(self.bricks)), "V": np.array(self.V),
                 "D_brick": np.array(self.D_brick), "seed": np.array(self.seed),
                 "neg_n": np.array(self.neg.n), "neg_sum": self.neg.sum, "neg_sumsq": self.neg.sumsq,
                 "doc_freq": self.doc_freq, "n_docs": np.array(self.n_docs),
                 "global_step": np.array(self._global_step)}
        for i, (brick, pos) in enumerate(zip(self.bricks, self.brick_pos)):
            bstate = brick.export_state()
            for k, v in bstate.items():
                state[f"brick{i}__{k}"] = v
            state[f"pos{i}__n"] = np.array(pos.n)
            state[f"pos{i}__sum"] = pos.sum
            state[f"pos{i}__sumsq"] = pos.sumsq
            # Loop-proof relay queue (2026-10-28): the cluster owns this (not the brick) since the
            # visited-history needs sibling visibility. Flattened the same way bricks/pos already
            # are -- one count key plus per-packet b/bg/visited keys.
            packets = brick.pending_relay
            visited_list = self.pending_visited[i]
            state[f"relay{i}__n"] = np.array(len(packets))
            for j, ((b_c, bg_c), vis) in enumerate(zip(packets, visited_list)):
                state[f"relay{i}__{j}__b"] = b_c
                state[f"relay{i}__{j}__bg"] = bg_c
                state[f"relay{i}__{j}__visited"] = np.array(sorted(vis), dtype=np.int64)
            # Verbatim taught-text store (2026-10-29, see brick_taught_lines' own field docstring)
            # -- same flatten-by-index pattern as the relay queue above, so retrieval survives a
            # real save/load cycle instead of silently resetting to empty every server restart.
            taught_lines = self.brick_taught_lines[i] if i < len(self.brick_taught_lines) else []
            state[f"taught{i}__n"] = np.array(len(taught_lines))
            for j, ln in enumerate(taught_lines):
                state[f"taught{i}__{j}"] = np.array(ln)
            # Confirmation-gated vote mechanism (2026-11-07 rebuild, see VOTE_MAX_VOTES's module
            # comment) -- same flatten-by-index pattern as taught/relay above.
            state[f"vote{i}__count"] = np.array(self.vote_count[i])
            state[f"vote{i}__last_confirm_step"] = np.array(self.last_confirm_step[i])
            state[f"vote{i}__ceiling_reached"] = np.array(self.ceiling_reached[i])
            # Usage-boost + challenger-must-prove-itself (2026-11-08 rebuild) -- same
            # flatten-by-index pattern as vote/taught/relay above.
            state[f"vote{i}__last_usage_step"] = np.array(self.last_usage_step[i])
            state[f"vote{i}__challenger_line"] = np.array(self.challenger_line[i])
            state[f"vote{i}__challenger_vote"] = np.array(self.challenger_vote[i])
        return state

    @classmethod
    def load_state(cls, state):
        n_bricks = int(state["n_bricks"])
        V = int(state["V"]); D_brick = int(state["D_brick"]); seed = int(state["seed"])
        state_keys = set(state.files) if hasattr(state, "files") else set(state.keys())
        # FIX (2026-11-05, real RAM-blowout root cause -- see PrivateLucyWEST's same fix /
        # BRICKCLUSTER_RAM_BLOWOUT_REPAIR_20261105.md for the full measured writeup): NpzFile's
        # .items()/.keys() are lazy with ZERO caching -- every call re-decompresses EVERY array in
        # the whole archive. The loop below calls state.items() once PER BRICK, so loading n_bricks
        # bricks did n_bricks full-archive decompressions (measured elsewhere: 402 bricks x 25,350
        # keys = 10.19M redundant reads for one checkpoint). Materializing once here turns that into
        # one real pass, total. No-op (shallow copy) if state is already a plain dict.
        state = dict(state.items())
        cluster = cls(V, D_brick, seed)
        cluster.neg.n = int(state["neg_n"])
        cluster.neg.sum = state["neg_sum"]
        cluster.neg.sumsq = state["neg_sumsq"]
        cluster.doc_freq = state["doc_freq"]
        cluster.n_docs = int(state["n_docs"])
        # Backward-compatible default (0) for any checkpoint saved before the confirmation-gated
        # vote mechanism existed -- global_step starting at 0 on a pre-existing checkpoint just
        # means decay math treats every brick as "freshly confirmed," the same safe default as a
        # brand-new cluster; it does not retroactively flip or decay anything.
        cluster._global_step = int(state["global_step"]) if "global_step" in state_keys else 0
        for i in range(n_bricks):
            brick = FixedStepBrick.__new__(FixedStepBrick)
            brick.V = V; brick.D = D_brick
            brick._res = {}; brick._mom = {}; brick._total_in = {}; brick._total_consumed = {}
            prefix = f"brick{i}__"
            bstate = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            brick.load_state(bstate)
            cluster.bricks.append(brick)
            pos = _OnlineGaussian()
            pos.n = int(state[f"pos{i}__n"])
            pos.sum = state[f"pos{i}__sum"]
            pos.sumsq = state[f"pos{i}__sumsq"]
            cluster.brick_pos.append(pos)
            # Loop-proof relay queue: backward-compatible default (empty) for any checkpoint saved
            # before this existed, including ones saved under the OLD (pre-redesign) single-array
            # pending_relay_b/bg scheme -- that scheme was ported this session but never actually
            # written to a real checkpoint (only exercised in throwaway smoke-test state), so no
            # real checkpoint needs migrating from it, only "never had a relay queue at all."
            n_relay_key = f"relay{i}__n"
            n_relay = int(state[n_relay_key]) if n_relay_key in state_keys else 0
            packets, visited_list = [], []
            for j in range(n_relay):
                packets.append((state[f"relay{i}__{j}__b"], state[f"relay{i}__{j}__bg"]))
                visited_list.append(frozenset(int(x) for x in state[f"relay{i}__{j}__visited"]))
            brick.pending_relay = packets
            cluster.pending_visited.append(visited_list)
            # Verbatim taught-text store: backward-compatible default (empty) for any checkpoint
            # saved before this existed -- INCLUDING the real, currently-live checkpoint
            # (amelia_teaches_lucy_mind.npz, 149 bricks saved before tonight's retrieval feature)
            # -- this is the exact bug a live test caught: without this branch,
            # cluster.brick_taught_lines stayed length 0 after loading 149 real bricks, and the
            # very next learn_line() call IndexError'd on brick_taught_lines[idx]. Found via a real
            # /v1/feedback call against the live server, not a toy harness (every toy test tonight
            # built a fresh BrickCluster() directly and never exercised load_state at all).
            n_taught_key = f"taught{i}__n"
            n_taught = int(state[n_taught_key]) if n_taught_key in state_keys else 0
            cluster.brick_taught_lines.append(
                [str(state[f"taught{i}__{j}"]) for j in range(n_taught)])
            # Confirmation-gated vote mechanism: backward-compatible default (vote=1, unconfirmed,
            # ceiling never reached) for any checkpoint saved before this existed -- same "never
            # had a vote at all" default as the real pre-existing checkpoint this rebuild has to
            # load, not a migration from some other prior scheme.
            vote_key = f"vote{i}__count"
            if vote_key in state_keys:
                cluster.vote_count.append(int(state[vote_key]))
                cluster.last_confirm_step.append(int(state[f"vote{i}__last_confirm_step"]))
                cluster.ceiling_reached.append(bool(state[f"vote{i}__ceiling_reached"]))
            else:
                cluster.vote_count.append(1)
                cluster.last_confirm_step.append(cluster._global_step)
                cluster.ceiling_reached.append(False)
            # Usage-boost + challenger-must-prove-itself: backward-compatible default (never used,
            # no pending challenger) for any checkpoint saved before this existed -- same "never had
            # this field at all" default pattern as vote_count's own backward-compat branch above.
            usage_key = f"vote{i}__last_usage_step"
            if usage_key in state_keys:
                cluster.last_usage_step.append(int(state[usage_key]))
                cluster.challenger_line.append(str(state[f"vote{i}__challenger_line"]))
                cluster.challenger_vote.append(int(state[f"vote{i}__challenger_vote"]))
            else:
                cluster.last_usage_step.append(-1)
                cluster.challenger_line.append("")
                cluster.challenger_vote.append(0)
        return cluster
