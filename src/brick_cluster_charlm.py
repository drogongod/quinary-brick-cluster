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


FEATURE_DIM = 64


def det_hash(s):
    """Deterministic string hash (Python's builtin hash() is salted per-process) -- reproducible
    across a save/load/restart cycle, unlike the builtin."""
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def line_features(line, dim=FEATURE_DIM):
    v = np.zeros(dim)
    s = line.lower()
    for i in range(len(s) - 2):
        v[det_hash(s[i:i + 3]) % dim] += 1
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class _OnlineGaussian:
    """Exact incremental diagonal-covariance Gaussian -- count/sum/sum-of-squares per dimension,
    so mean/var are always exact (not an EMA approximation) and cost O(dim) per update, no refit."""
    def __init__(self, dim=FEATURE_DIM):
        self.n = 0
        self.sum = np.zeros(dim)
        self.sumsq = np.zeros(dim)

    def update(self, x):
        self.n += 1
        self.sum += x
        self.sumsq += x * x

    def mean_var(self, var_floor_frac, global_var):
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
        return mean, np.maximum(var, floor)

    def log_density(self, x, var_floor_frac, global_var):
        mean, var = self.mean_var(var_floor_frac, global_var)
        return -0.5 * np.sum((x - mean) ** 2 / var + np.log(2 * np.pi * var))


VAR_FLOOR_FRAC = 0.05

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
MERGE_SLACK = 3.0   # log-density units; margin > -MERGE_SLACK accepted. See sweep above for the
# full curve before changing this -- re-verify against a real sweep, not a guess, if ever revisited.
MERGE_MAX_ROUNDS = 10   # safety cap on the fixed-point iteration -- merges strictly reduce brick
# count each round so termination is guaranteed regardless, this just bounds worst-case cost.


class BrickCluster:
    """Growable cluster of FixedStepBrick, routed by an online LSL-style gate. V/D_brick fixed at
    construction (every brick the same size); the NUMBER of bricks grows as genuinely new content
    arrives that no existing brick's Phi_pos explains better than the shared Phi_neg."""
    def __init__(self, V, D_brick, seed):
        self.V = V
        self.D_brick = D_brick
        self.seed = seed
        self.bricks = []
        self.brick_pos = []       # one _OnlineGaussian per brick (Phi_pos)
        self.neg = _OnlineGaussian()   # shared Phi_neg, every line ever seen

    def _global_var(self):
        if self.neg.n < 2:
            return np.ones(FEATURE_DIM)
        mean = self.neg.sum / self.neg.n
        return np.maximum(self.neg.sumsq / self.neg.n - mean * mean, 1e-6)

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

    def learn_line(self, batches_for_line, features, lr_scale):
        """batches_for_line: the windowed batches (same bucket-by-length convention as
        lutgtlm_ingest.py) for ONE line. Routes (possibly spawning), trains, updates the gate."""
        idx = self.route(features, for_training=True)
        brick = self.bricks[idx]
        losses = []
        for b in batches_for_line:
            losses.append(float(brick.learn_batch(b, lr_scale=lr_scale).mean()))
        self.brick_pos[idx].update(features)
        self.neg.update(features)
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
                for big_i in order[:rank]:   # only consider STRICTLY bigger/equal-or-earlier bricks
                    if big_i in absorbed_into or big_i == small_i:
                        continue
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

    def forward_logits(self, idxs, features):
        idx = self.route(features, for_training=False)
        if idx is None:
            return None
        return self.bricks[idx].forward_logits(idxs)

    def generate(self, seed_idx, itos, features, n=60, temp=0.7, rep_pen=1.0, seed=0):
        idx = self.route(features, for_training=False)
        if idx is None:
            return ""
        return self.bricks[idx].generate(seed_idx, itos, n=n, temp=temp, rep_pen=rep_pen, seed=seed)

    def export_state(self):
        state = {"n_bricks": np.array(len(self.bricks)), "V": np.array(self.V),
                 "D_brick": np.array(self.D_brick), "seed": np.array(self.seed),
                 "neg_n": np.array(self.neg.n), "neg_sum": self.neg.sum, "neg_sumsq": self.neg.sumsq}
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
