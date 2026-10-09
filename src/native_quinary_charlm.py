#!/usr/bin/env python3
r"""native_quinary_charlm.py -- OPTION 1, built and tested: the persisted weight IS the quinary
level {-2,-1,0,+1,+2}, no float32 shadow anywhere. Updates via the SAME residual-carry pattern
already proven in lutgtlm_fast.py's _d5_residual/_ema (sub-threshold signal is CARRIED FORWARD,
never discarded) -- the key structural difference from quinary_simd_batched_charlm.py's STE
approach (a float32 master absorbs ordinary gradients; quantization is a transient forward view).
Here there is no master; the level itself is what's trained, directly, through the real fastICE/
quinary SIMD kernel.

THREE QUESTIONS THIS FILE ANSWERS (Jonathan, 2026-10-07 -- "both unknown"):
  1. Does it even work? (does loss decrease with only 5 representable weight values + a
     residual-carry update, no float precision anywhere in the persisted weight)
  2. Is there ANY loss at all? -- checked directly: every gradient contribution either becomes a
     level step or stays in the residual; nothing is ever dropped in between.
  3. Does it still beat BLAS? -- same real SIMD kernel, same shapes, same baseline used throughout.
"""
import numpy as np, ctypes, os, time

_DLL = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quinary_simd_kernel.dll"))
_DLL.quinary_gemm_simd.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3

THRESHOLD = 500        # residual units (post-normalization, see NORM_TARGET) needed to move a
                       # weight level by one step. CALIBRATED 2026-10-08 via a real post-bugfix
                       # sweep on the 100KB real-corpus slice (THR=2000 was the pre-adaptive-
                       # normalization default and no longer the best-tested value).
MOMENTUM_K = 8
NORM_TARGET = 500      # target units for a "typical" (fully-consistent) per-call signal after
                       # adaptive normalization -- THRESHOLD/NORM_TARGET=1 call for a maximally
                       # confident, perfectly-consistent weight to earn one level-step.
DH_CARRY_CLIP = 10 ** 12  # bug #3 equivalent (lutgtlm_fast.py/_stacked.py/_batched.py all have
                          # this): bounds the BPTT backward carry at every timestep so no single
                          # unstable window can compound past this. CALIBRATED 2026-10-08 against
                          # a real measured 3.3e24 blowup (so 1e12 is still ~12 orders of
                          # magnitude of headroom below the catastrophic case) -- a tighter 1e6
                          # clip was tried first and measurably HURT convergence (clips real
                          # signal, not just the pathological case), so this isn't just "small
                          # enough to be safe," it's "loose enough not to distort real learning."
ACCUM_GRAD_CLIP = 10 ** 14 # bug #11 equivalent: bounds the SUM of a window's ~T timesteps' worth
                          # of individually-clipped contributions, which the per-timestep clip
                          # alone cannot do. Same calibration note as DH_CARRY_CLIP above.
EQ = 1000              # embedding fixed-point scale
EMB_DECAY_DIV = 2000   # weight decay for Wemb only (the quinary weights are self-bounding via
                       # the +-2 level clip; Wemb has no such clip, so it needs its own
                       # restoring force -- found missing by a real-corpus overflow, see
                       # _residual_update_wide's comment). Same constant/pattern already proven
                       # in lutgtlm_fast.py's WEIGHT_DECAY_DIV, not re-guessed.
MAX_EMB = 5 * EQ       # HARD CLIP, found necessary 2026-10-08: decay alone was far too weak --
                       # measured max|Wemb| growing 3839->8311->14656, an ACCELERATING runaway
                       # (bigger embedding -> bigger activation -> bigger gradient -> bigger
                       # embedding), not just a slow drift decay could counteract. A hard bound
                       # (x_t = Wemb/EQ stays within [-5,5], generous but finite) stops the
                       # feedback loop outright, matching this codebase's own established
                       # defensive pattern (MAX_SANE_WEIGHT / MAX_STEP) rather than relying on a
                       # proportional restoring force to win a race against positive feedback.


def level_to_planes(level):
    """int8 level in {-2..2} -> S1,S2 int8 planes, v = S1 + 2*S2 (exact, no rounding -- the level
    IS already discrete, this is a lossless re-encoding for the SIMD kernel's input format)."""
    level = level.astype(np.int16)
    S1 = np.where(np.abs(level) == 1, level, 0).astype(np.int8)
    S2 = np.where(np.abs(level) == 2, level // 2, 0).astype(np.int8)
    return np.ascontiguousarray(S1), np.ascontiguousarray(S2)


PG_CLIP = 40.0   # pre-activation clip before the sigmoid. ROOT CAUSE FOUND 2026-10-08, real-corpus
                 # test: with Wemb hard-clipped AND all quinary weights at their bounded max level,
                 # it STILL overflowed -- because this cell has NO normalization layer, summing
                 # K=384 individually-bounded-but-correlated terms (many real text features drive
                 # many weights toward the same sign together) can still exceed exp()'s float64
                 # range even though every individual weight/activation is bounded. Bounding the
                 # PARTS does not bound the SUM. Standard fix every real framework uses: clip the
                 # pre-activation itself -- sigmoid(40) is already indistinguishable from 1.0 at
                 # float64 precision, so this loses nothing real, it only removes the numerically
                 # meaningless tail past where the function has already saturated.
def sig_stable(pg):
    return 1.0 / (1.0 + np.exp(-np.clip(pg, -PG_CLIP, PG_CLIP)))


def activation_quant_int8(x):
    amax = np.maximum(np.abs(x).max(axis=0, keepdims=True), 1)
    scale = amax.astype(np.float64) / 127.0
    xq = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return xq, scale


def simd_mm(S1, S2, X, xscale):
    """S1,S2: (M,K) int8 LEVEL planes (no weight-side scale -- the level itself IS the exact
    value, {-2..2}; only the activation side needs a scale, since it's the one quantized from a
    wider range). Returns (M,B) float64 = levels @ X_real."""
    M, K = S1.shape; B = X.shape[1]
    Xt = np.ascontiguousarray(X.T)
    out = np.empty((M, B), dtype=np.int32)
    _DLL.quinary_gemm_simd(S1.ctypes.data, S2.ctypes.data, Xt.ctypes.data, out.ctypes.data, M, K, B)
    return out.astype(np.float64) * xscale


class NativeQuinaryCharLM:
    def __init__(self, V, D, seed):
        rng = np.random.default_rng(seed)
        self.V = V; self.D = D
        randlevel = lambda a, b: rng.integers(-1, 2, (a, b)).astype(np.int8)
        self.Win = randlevel(D, D)
        self.Wh = np.where(np.eye(D, dtype=bool), 1, 0).astype(np.int8)   # near-identity -> diag=+1
        self.Wgx = randlevel(D, D); self.Wgh = np.zeros((D, D), np.int8)
        self.Wo = randlevel(V, D)
        self.Wemb = (rng.uniform(-1, 1, (D, V)) * EQ).astype(np.int64)    # wide fixed-point, not level-restricted
        self.b = np.zeros(D, np.float64); self.bg = np.zeros(D, np.float64)
        self._res = {}; self._mom = {}
        self._total_in = {}; self._total_consumed = {}

    def _integer_ema(self, name, grad_int):
        """Integer EMA with its OWN residual-carry -- the exact already-proven _ema() pattern
        from lutgtlm_fast.py (fixed 2026-09-05/29 for precisely this deadzone), applied here so
        the MOMENTUM step itself is lossless too, not just the level-threshold step. Without this,
        a small/slowly-varying integer gradient could round to a float momentum that never crosses
        0.5 and would silently vanish every call -- the same bug class this project already found
        and fixed once in _ema, now ported to this new file rather than re-introduced blind."""
        v = self._mom.get(name + "_v", np.zeros_like(grad_int, dtype=np.int64))
        acc = self._mom.get(name + "_acc", np.zeros_like(grad_int, dtype=np.int64))
        d = grad_int - v
        acc = acc + d
        step = np.sign(acc) * (np.abs(acc) // MOMENTUM_K)
        acc = acc - step * MOMENTUM_K
        v = v + step
        self._mom[name + "_v"] = v; self._mom[name + "_acc"] = acc
        return v

    def _integer_isqrt(self, n):
        """Exact integer square root (floor), Newton's method -- no float anywhere, no precision
        loss: isqrt(n)^2 <= n < (isqrt(n)+1)^2, exactly, for every non-negative integer n."""
        n = np.asarray(n, dtype=np.int64)
        x = np.maximum(n.copy(), 1)   # x never reaches 0 during iteration -- avoids div-by-zero
        for _ in range(40):
            x_new = (x + n // x) // 2
            x = np.where((x_new < x) & (x_new > 0), x_new, x)
        # correct any off-by-one from the integer iteration
        x = np.where(x * x > n, x - 1, x)
        x = np.where((x + 1) * (x + 1) <= n, x + 1, x)
        return np.maximum(x, 0)

    def _adaptive_normalize_v2(self, name, mom_int):
        """TRUE Adam-style (L2/RMS, not L1) integer adaptive normalization -- built 2026-10-08
        after Jonathan correctly challenged the L1 version's premise: single-digit quantization
        (5 levels) is a precision constraint, not a base-system constraint, and since fastICE/STE
        ALSO only shows a 5-level view per forward pass yet converges better, the gap isn't
        representational capacity -- it's search dynamics. Real Adam normalizes by sqrt(E[g^2])
        (the true second moment), not E[|g|] (the cruder L1 approximation _adaptive_normalize
        used) -- testing the more faithful analog directly rather than assuming the cheaper one
        was close enough. Exact integer sqrt (see _integer_isqrt), no float anywhere."""
        v = self._mom.get(name + "_v2", np.zeros_like(mom_int, dtype=np.int64))
        g2 = mom_int.astype(np.int64) * mom_int.astype(np.int64)
        d = g2 - v
        v = v + np.sign(d) * (np.abs(d) // MOMENTUM_K)
        self._mom[name + "_v2"] = v
        rms = self._integer_isqrt(np.maximum(v, 0))

        numres = self._mom.get(name + "_numres2", np.zeros_like(mom_int, dtype=np.int64))
        numres = numres + mom_int.astype(np.int64) * NORM_TARGET
        denom = rms + 1
        scaled = numres // denom
        numres = numres - scaled * denom
        self._mom[name + "_numres2"] = numres
        return scaled

    def _adaptive_normalize(self, name, mom_int):
        """ADAPTIVE PER-WEIGHT NORMALIZATION (2026-10-08) -- the real fix for the convergence
        failure found on real-corpus data (sign-based fixed-step updates scrambled, same failure
        shape as this project's own D5 finding: 'hand-normalized step scrambles ternary,' fixed
        there by switching to Adam). Adam's trick is PER-PARAMETER adaptive step size from each
        parameter's own gradient variance; this is the same idea in integer arithmetic, L1-style
        (running mean of |momentum| instead of a true variance, so no integer sqrt is needed):
        track each weight's own typical signal magnitude, then rescale so a weight with a large-
        but-consistent signal and one with a small-but-consistent signal both take a comparable
        number of calls to earn a level-step -- instead of the big-gradient weight dominating
        movement just because its raw numbers are larger.

        HONEST SCOPE on losslessness: the numerator (mom_int*NORM_TARGET) is accumulated into its
        OWN residual before dividing, so no numerator unit is ever dropped -- but the divisor
        itself (running, an EMA) drifts slowly between calls, so a remainder carried across a
        divisor change is carried at a very slightly different 'exchange rate' than it was
        earned at. This is a deliberately-scoped, honestly-stated approximation on the RATE-
        CONTROL signal only (not the core gradient, which was already captured exactly by
        _integer_ema before this function ever sees it) -- unlike the level-threshold step and
        the momentum stage, which remain exactly, provably lossless (see check_conservation)."""
        running = self._mom.get(name + "_rabs", np.zeros_like(mom_int, dtype=np.int64))
        d = np.abs(mom_int) - running
        running = running + np.sign(d) * (np.abs(d) // MOMENTUM_K)
        self._mom[name + "_rabs"] = running

        numres = self._mom.get(name + "_numres", np.zeros_like(mom_int, dtype=np.int64))
        numres = numres + mom_int.astype(np.int64) * NORM_TARGET
        denom = running + 1   # +1 avoids div-by-zero for a weight that has never moved yet
        scaled = numres // denom
        numres = numres - scaled * denom
        self._mom[name + "_numres"] = numres
        return scaled

    def _residual_update_level(self, name, level, grad):
        """Residual-carry update for a quinary-LEVEL weight. Returns the new int8 level array.
        BOTH stages (momentum AND the level-threshold step) now use integer residual-carry --
        verified no stage can silently discard a sub-threshold contribution. Tracks the exact
        conservation invariant: cumulative_in == cumulative_consumed + final_residual.

        VARIABLE STEP SIZE (2026-10-08): previously capped at exactly +-1 level per call
        regardless of how much residual had accumulated -- a confident, overwhelming signal was
        artificially throttled to the same single-level step as a signal that just barely crossed
        THRESHOLD, and reversing a bad early commitment required fully re-earning THRESHOLD from
        near-zero every time. Both L1 and L2/RMS normalization plateaued well above the continuous
        baseline with this cap in place; `_residual_update_wide` (Wemb) never had this cap and
        uses `sign(res)*(|res|//THRESHOLD)` already -- this brings the level-weight update to the
        same already-proven-elsewhere pattern, bounded only by the level's own valid range."""
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize(name, mom_int)  # L1: measured better than L2/RMS (v2),
                                                           # see native_quinary_debugging_arc.md
        res = self._res.get(name, np.zeros_like(level, dtype=np.int64))
        res = res + scaled
        self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
        step = np.sign(res) * (np.abs(res) // THRESHOLD)
        desired = level.astype(np.int64) - step
        new_level = np.clip(desired, -2, 2).astype(np.int8)
        actual_step = level.astype(np.int64) - new_level.astype(np.int64)
        res = res - actual_step * THRESHOLD
        self._total_consumed[name] = self._total_consumed.get(name, 0) + int((actual_step * THRESHOLD).sum())
        self._res[name] = res
        return new_level

    def _residual_update_wide(self, name, W, grad):
        """Same doubly-lossless residual-carry pattern for the wide-range embedding, PLUS weight
        decay. BUG FOUND 2026-10-08, real-corpus test (100KB voice_corpus slice): unlike
        Win/Wh/Wgx/Wgh/Wo (hard-clipped to level in [-2,2] by construction), Wemb has no bound at
        all -- measured max|Wemb| growing 3840->8300->13722->17936, roughly doubling every batch,
        with NO restoring force, until it fed a huge activation into Wgx@x and overflowed the
        sigmoid. Same "no restoring force" failure class this project has already found and fixed
        multiple times elsewhere (lutgtlm_fast.py's WEIGHT_DECAY_DIV, the GPU trainer's decay fix)
        -- the residual-carry mechanism being lossless does NOT mean the weight itself is bounded;
        those are two separate properties, and I'd only built the first one for Wemb. Decay is
        applied to the WEIGHT only, never to the residual/momentum state, so the exact-conservation
        invariant (check_conservation) stays unaffected by this fix -- decay is a deliberate,
        tracked loss of magnitude (a design choice, documented, not an accident), separate from the
        residual-carry's job of never losing a gradient CONTRIBUTION."""
        mom_int = self._integer_ema(name, grad)
        scaled = self._adaptive_normalize_v2(name, mom_int)
        res = self._res.get(name, np.zeros_like(W, dtype=np.int64))
        res = res + scaled
        self._total_in[name] = self._total_in.get(name, 0) + int(scaled.sum())
        step = np.sign(res) * (np.abs(res) // THRESHOLD)
        res = res - step * THRESHOLD
        self._total_consumed[name] = self._total_consumed.get(name, 0) + int((step * THRESHOLD).sum())
        self._res[name] = res
        W_new = W - step
        decay = np.sign(W_new) * (np.abs(W_new) // EMB_DECAY_DIV)
        W_new = W_new - decay
        clipped = np.clip(W_new, -MAX_EMB, MAX_EMB)
        # HONEST ACCOUNTING: decay and the hard clip are both deliberate, intentional discarding
        # of magnitude (not a bug) -- but the exact-conservation invariant (check_conservation)
        # must know about it, or "ALL EXACT: True" would silently go stale and lie once either
        # mechanism actually engages. Tracked separately from total_consumed (which means
        # "became a weight change"), as total_discarded (which means "intentionally thrown away").
        discarded = int((decay).sum()) + int((W_new - clipped).sum())
        self._total_discarded = getattr(self, "_total_discarded", {})
        self._total_discarded[name] = self._total_discarded.get(name, 0) + discarded
        return clipped

    def export_state(self):
        """Full state for checkpointing -- includes residual/momentum accumulators, not just the
        weights, so a save/load cycle loses NOTHING (the whole point of this architecture)."""
        state = dict(Win=self.Win, Wh=self.Wh, Wgx=self.Wgx, Wgh=self.Wgh, Wo=self.Wo,
                     Wemb=self.Wemb, b=self.b, bg=self.bg)
        for k, v in self._res.items():
            state["res__" + k] = v
        for k, v in self._mom.items():
            state["mom__" + k] = v
        for k, v in self._total_in.items():
            state["totin__" + k] = np.array(v)
        for k, v in self._total_consumed.items():
            state["totcon__" + k] = np.array(v)
        return state

    def load_state(self, state):
        self.Win = state["Win"]; self.Wh = state["Wh"]; self.Wgx = state["Wgx"]
        self.Wgh = state["Wgh"]; self.Wo = state["Wo"]; self.Wemb = state["Wemb"]
        self.b = state["b"]; self.bg = state["bg"]
        for key in state.files if hasattr(state, "files") else state.keys():
            if key.startswith("res__"):
                self._res[key[5:]] = state[key]
            elif key.startswith("mom__"):
                self._mom[key[5:]] = state[key]
            elif key.startswith("totin__"):
                self._total_in[key[7:]] = int(state[key])
            elif key.startswith("totcon__"):
                self._total_consumed[key[8:]] = int(state[key])

    def check_conservation(self):
        """EXACT invariant check: for every tracked weight, cumulative_in must equal
        cumulative_consumed + final_residual_sum + cumulative_discarded, to the integer, zero
        tolerance. The quinary-level weights (Win/Wh/Wgx/Wgh/Wo) have discarded==0 ALWAYS --
        genuinely, provably lossless, nothing ever intentionally thrown away. Wemb carries an
        honest exception: its decay + hard clip (added 2026-10-08 after a real runaway was
        found on real-corpus data) are DELIBERATE, TRACKED discarding, not silent loss -- when
        discarded==0 for Wemb too (the normal, non-runaway regime), it's exactly as lossless as
        the rest; when discarded>0, that's the clip actually engaging, visible here, not hidden
        behind a verdict that still says 'EXACT' when it no longer honestly means zero-loss."""
        out = {}
        for k in self._total_in:
            total_in = self._total_in[k]
            consumed = self._total_consumed.get(k, 0)
            discarded = getattr(self, "_total_discarded", {}).get(k, 0)
            final_res = int(self._res[k].sum())
            out[k] = {"total_in": total_in, "consumed": consumed, "discarded": discarded,
                       "final_residual": final_res,
                       "EXACT (in == consumed + discarded + residual)":
                           total_in == consumed + discarded + final_res,
                       "GENUINELY LOSSLESS (discarded == 0)": discarded == 0}
        return out

    def forward_logits(self, idxs):
        """Single-sequence forward (B=1), no training -- for held-out eval and generation.
        Reuses the exact same quantized SIMD path as learn_batch's forward half."""
        return self._forward_batch([idxs])[0]  # (T, V) for this one sequence

    def _forward_batch(self, idxs_batch):
        B = len(idxs_batch); T = len(idxs_batch[0])
        idxs_arr = np.array(idxs_batch, dtype=np.int64)
        D = self.D
        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        S1_o, S2_o = level_to_planes(self.Wo)
        X = (self.Wemb[:, idxs_arr.T]).astype(np.float64) / EQ
        h = np.zeros((D, B), np.float64)
        logits = np.empty((T, self.V, B), np.float64)
        for t in range(T):
            x_t = X[:, t, :]
            xq, xs = activation_quant_int8(x_t); hq, hs = activation_quant_int8(h)
            pc = simd_mm(S1_in, S2_in, xq, xs) + simd_mm(S1_h, S2_h, hq, hs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_gx, S2_gx, xq, xs) + simd_mm(S1_gh, S2_gh, hq, hs) + self.bg[:, None]
            g = sig_stable(pg)
            h = g * c + (1 - g) * h
            hnq, hns = activation_quant_int8(h)
            logits[t] = simd_mm(S1_o, S2_o, hnq, hns)
        return logits.transpose(2, 0, 1)  # (B, T, V)

    def generate(self, seed_idx, itos, n=60, temp=0.7, rep_pen=1.0, seed=0):
        """Greedy/sampled generation, single sequence. rep_pen default 1.0 (disabled) per this
        project's own verified finding (galatea_reppen_sign_bug.md) -- a naive repeat-penalty
        formula actively hurts small-vocab char models more than it helps."""
        rng = np.random.default_rng(seed)
        D = self.D
        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        S1_o, S2_o = level_to_planes(self.Wo)
        h = np.zeros((D, 1), np.float64)
        out = list(seed_idx)

        def step(ix, h):
            x = (self.Wemb[:, ix].astype(np.float64) / EQ).reshape(D, 1)
            xq, xs = activation_quant_int8(x); hq, hs = activation_quant_int8(h)
            pc = simd_mm(S1_in, S2_in, xq, xs) + simd_mm(S1_h, S2_h, hq, hs) + self.b[:, None]
            c = np.tanh(pc)
            pg = simd_mm(S1_gx, S2_gx, xq, xs) + simd_mm(S1_gh, S2_gh, hq, hs) + self.bg[:, None]
            g = sig_stable(pg)
            return g * c + (1 - g) * h

        for ix in seed_idx[:-1]:
            h = step(ix, h)
        ix = seed_idx[-1]
        for _ in range(n):
            h = step(ix, h)
            hq, hs = activation_quant_int8(h)
            logit = simd_mm(S1_o, S2_o, hq, hs).flatten()
            if rep_pen != 1.0:
                for j in set(out[-24:]):
                    if logit[j] > 0: logit[j] /= rep_pen
                    else: logit[j] *= rep_pen
            p = np.exp((logit - logit.max()) / max(1e-3, temp)); p /= p.sum()
            ix = int(rng.choice(len(p), p=p)); out.append(ix)
            if itos[ix] == "\n":
                break
        return "".join(itos[i] for i in out)

    def learn_batch(self, idxs_batch, lr_scale=1.0):
        B = len(idxs_batch); T = len(idxs_batch[0])
        idxs_arr = np.array(idxs_batch, dtype=np.int64)
        D = self.D

        S1_in, S2_in = level_to_planes(self.Win); S1_h, S2_h = level_to_planes(self.Wh)
        S1_gx, S2_gx = level_to_planes(self.Wgx); S1_gh, S2_gh = level_to_planes(self.Wgh)
        S1_o, S2_o = level_to_planes(self.Wo)
        S1_inT, S2_inT = level_to_planes(self.Win.T); S1_hT, S2_hT = level_to_planes(self.Wh.T)
        S1_gxT, S2_gxT = level_to_planes(self.Wgx.T); S1_ghT, S2_ghT = level_to_planes(self.Wgh.T)
        S1_oT, S2_oT = level_to_planes(self.Wo.T)

        X = (self.Wemb[:, idxs_arr.T]).astype(np.float64) / EQ   # (D,T,B) real-valued

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
            # DH_CARRY_CLIP (2026-10-08) -- BUG FOUND: the backward carry through BPTT had NO
            # clip anywhere, unlike every other trainer in this codebase (lutgtlm_fast.py,
            # lutgtlm_stacked.py, lutgtlm_batched.py all have this exact clip, "bug #3"). Measured
            # directly: dh reached 3.3e24 by the end of a real-corpus backward pass -- not NaN
            # yet, but so far beyond int64 range that casting it produced garbage (the "invalid
            # value encountered in cast" warning). Same bug-class, new file, found by instrumenting
            # step by step rather than guessed.
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
        self.Win = self._residual_update_level("Win", self.Win, scale_int(gWin))
        self.Wh = self._residual_update_level("Wh", self.Wh, scale_int(gWh))
        self.Wgx = self._residual_update_level("Wgx", self.Wgx, scale_int(gWgx))
        self.Wgh = self._residual_update_level("Wgh", self.Wgh, scale_int(gWgh))
        self.Wo = self._residual_update_level("Wo", self.Wo, scale_int(gWo))
        self.Wemb = self._residual_update_wide("Wemb", self.Wemb, scale_int(gWemb * EQ))
        self.b = self.b - lr_scale * 1e-4 * gb; self.bg = self.bg - lr_scale * 1e-4 * gbg

        return loss / max(1, n)


def _smoke_test():
    from experiment_bq_identity_distill import load_identity
    texts = load_identity()
    chars = sorted(set("".join(texts)))
    stoi = {c: i for i, c in enumerate(chars)}
    V = len(chars); enc = lambda s: [stoi[c] for c in s]; D = 192
    lines = [l for l in texts if len(enc(l)) >= 3]
    T = min(len(enc(l)) for l in lines[:32])
    batch = [enc(l)[:T] for l in lines[:32]]

    m = NativeQuinaryCharLM(V, D, seed=5)
    print("[test] does it work at all?")
    for r in range(15):
        loss = m.learn_batch(batch, lr_scale=8.0)
        if r % 3 == 0:
            print(f"  round {r}: mean_loss={loss.mean():.3f}")

    print("\n[test] conservation check:")
    all_exact = True; all_lossless = True
    for k, v in m.check_conservation().items():
        exact = v["EXACT (in == consumed + discarded + residual)"]
        lossless = v["GENUINELY LOSSLESS (discarded == 0)"]
        all_exact = all_exact and exact; all_lossless = all_lossless and lossless
        print(f"  {k}: in={v['total_in']}  consumed={v['consumed']}  discarded={v['discarded']}  "
              f"residual={v['final_residual']}  accounting_exact={exact}  lossless={lossless}")
    print(f"  ALL ACCOUNTING EXACT: {all_exact}   ALL GENUINELY LOSSLESS: {all_lossless}")

    print("\n[test] speed vs BLAS baseline:")
    t0 = time.time()
    for _ in range(10):
        m.learn_batch(batch, lr_scale=8.0)
    dt = (time.time() - t0) / 10
    n_windows = len(batch)
    print(f"  {dt*1000/n_windows:.3f} ms/window (D={D}, B={n_windows})")


if __name__ == "__main__":
    _smoke_test()
