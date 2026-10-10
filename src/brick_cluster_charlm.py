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
)


class FixedStepBrick(NativeQuinaryCharLM):
    """L1-normalize + fixed +-1-level-per-call step -- the best-tested single-brick config
    (native_quinary_debugging_arc.md), not the production file's current variable-step formula."""
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
MERGE_SLACK = 0.22   # mean-log-density units (see log_density's np.mean normalization); margin >
# -MERGE_SLACK accepted. This value is NOT comparable to the old sum-based sweep above -- re-
# verify against a real sweep, not a guess, if FEATURE_DIM or the log_density aggregation ever
# changes again (both silently rescale what this constant means, confirmed twice this session).
MERGE_MAX_ROUNDS = 10   # safety cap on the fixed-point iteration -- merges strictly reduce brick
# count each round so termination is guaranteed regardless, this just bounds worst-case cost.

MERGE_CANDIDATE_K = 10   # bound the number of bigger-brick candidates tested per merge decision --
# see the real multiple-comparisons bug this fixes in consolidate()'s own comment. Same role as
# WOLFPACK_GATE_K_NEIGHBORS in sovereign_core.py's own LSL-gate port.


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
        return idx

    def route(self, features, for_training):
        """Returns a brick index. TRAINING may spawn a new brick if nothing existing fits
        (Phi_pos_i - Phi_neg > 0 for no i); INFERENCE never spawns -- always returns the
        best-available brick (spawning on a read-only call would mean generating from an
        untrained net, which is never useful), matching LSL's own training/inference asymmetry."""
        if not self.bricks:
            return self._spawn() if for_training else None
        gvar = self._global_var()
        neg_ld = self.neg.log_density(features, VAR_FLOOR_FRAC, gvar) if self.neg.n >= 2 else -np.inf
        margins = [pos.log_density(features, VAR_FLOOR_FRAC, gvar) - neg_ld for pos in self.brick_pos]
        best_idx = int(np.argmax(margins))
        if for_training and margins[best_idx] <= 0.0:
            return self._spawn()
        return best_idx

    def learn_line(self, batches_for_line, line_text, lr_scale):
        """batches_for_line: the windowed batches (same bucket-by-length convention as
        lutgtlm_ingest.py) for ONE line. Routes (possibly spawning), trains, updates the gate.
        Takes the raw line TEXT (not pre-computed features, unlike the original version) so TF-IDF
        weighting can use the CURRENT doc-frequency stats before updating them with this line --
        computing features externally would make doc-frequency tracking inconsistent/bypassable."""
        features = self.tfidf_sparse(line_text)
        idx = self.route(features, for_training=True)
        brick = self.bricks[idx]
        losses = []
        for b in batches_for_line:
            losses.append(float(brick.learn_batch(b, lr_scale=lr_scale).mean()))
        self.brick_pos[idx].update(features)
        self.neg.update(features)
        self._update_doc_freq(line_text)
        return idx, (sum(losses) / len(losses) if losses else None)

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
        total_merged = 0
        for _round in range(MERGE_MAX_ROUNDS):
            if len(self.bricks) < 2:
                break
            order = sorted(range(len(self.bricks)), key=lambda i: self.brick_pos[i].n, reverse=True)
            gvar = self._global_var()
            neg_ld_cache = {}

            def neg_ld(x_key, x):
                if x_key not in neg_ld_cache:
                    neg_ld_cache[x_key] = self.neg.log_density(x, VAR_FLOOR_FRAC, gvar) if self.neg.n >= 2 else -np.inf
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
                        absorbed_into[small_i] = big_i
                        break

            if not absorbed_into:
                break
            keep = [i for i in range(len(self.bricks)) if i not in absorbed_into]
            self.bricks = [self.bricks[i] for i in keep]
            self.brick_pos = [self.brick_pos[i] for i in keep]
            total_merged += len(absorbed_into)
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
                 "doc_freq": self.doc_freq, "n_docs": np.array(self.n_docs)}
        for i, (brick, pos) in enumerate(zip(self.bricks, self.brick_pos)):
            bstate = brick.export_state()
            for k, v in bstate.items():
                state[f"brick{i}__{k}"] = v
            state[f"pos{i}__n"] = np.array(pos.n)
            state[f"pos{i}__sum"] = pos.sum
            state[f"pos{i}__sumsq"] = pos.sumsq
        return state

    @classmethod
    def load_state(cls, state):
        n_bricks = int(state["n_bricks"])
        V = int(state["V"]); D_brick = int(state["D_brick"]); seed = int(state["seed"])
        cluster = cls(V, D_brick, seed)
        cluster.neg.n = int(state["neg_n"])
        cluster.neg.sum = state["neg_sum"]
        cluster.neg.sumsq = state["neg_sumsq"]
        cluster.doc_freq = state["doc_freq"]
        cluster.n_docs = int(state["n_docs"])
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
        return cluster
