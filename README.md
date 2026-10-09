# Quinary Brick Cluster — Exactly-Lossless Continuous Learning via a Growable, Gated Substrate

A native **base-5 (quinary)** weight representation — the persisted weight *is* the level
`{-2,-1,0,+1,+2}`, no float shadow anywhere — trained via **residual-carry** (sub-threshold
gradient signal is carried forward, never discarded) through a real multiply-free SIMD kernel,
composed into a **growable cluster of small, fully-trained bricks** routed by a real, fitted
density gate instead of a fixed topology. Built as the continuous-learning counterpart to
[**lossless-ternary / fastICE**](https://github.com/drogongod/lossless-ternary): that repo is the
frozen/periodic-target tool (Option 2 — float32 master + STE, fast convergence, a small measured
standing bias); this one is the always-learning tool (Option 1 — no master copy, exactly lossless,
slower convergence per-pass, no bias to ever correct for).

## The result this exists to capture

A single large net trained on real text hits a real quality ceiling well short of proven
Adam-based continuous methods. Splitting the same capacity across many small, independently
*fully*-trained bricks — not frozen, not a reservoir — closes most of that gap:

| Substrate | Params | Loss (same corpus) |
|---|---|---|
| Single D=384 net, full BPTT | ~627K | 10.3–11.7 |
| 8 oracle-clustered bricks (k-means routing, known answer) | ~82K | 6.0–6.6 |
| Growable brick cluster, **real fitted gate, no oracle** | ~722K | held-out ~4.1–4.5 |

The middle row proves the substrate works when routing is given. The real engineering question —
does a *fitted* gate (no cheating) recover that, at inference time, on data it wasn't told the
answer for — is what this repo actually answers: **yes**, confirmed by swapping k-means's oracle
assignment for a real gate and measuring 94.2% routing agreement with the oracle, at equal or
slightly better loss.

## How the gate works

Adapted from **Local Support Learning** (Ben-Kish, Kumar, Glass, Giryes — Tel Aviv University /
MIT CSAIL, [arXiv:2610.02126](https://arxiv.org/abs/2610.02126)): each brick fits a diagonal
Gaussian (`Phi_pos`) over the content it's actually trained on; one shared Gaussian (`Phi_neg`) is
fit over everything ever seen, any brick. A line routes to whichever brick's `Phi_pos` beats the
shared background by the largest margin; if nothing does, a new brick spawns. Training always
knows its own data (route-or-spawn); inference never spawns — it always picks the best-available
brick, the same training/inference asymmetry LSL itself uses.

## Growth needs consolidation, and consolidation needs a measured safety margin

Online, one-line-at-a-time growth is greedy — a brick that claims a region early is never
revisited, so left alone it over-spawns relative to what oracle clustering finds (measured: 148
bricks for 224 lines, *more* total parameters than the single net this exists to beat). A periodic
re-merge pass fixes that: sort bricks by how much they've absorbed, and ask whether a bigger
brick's own `Phi_pos` already accepts a smaller brick's territory — same accept/reject mechanism
already proven for routing, reused for merging, no weight-averaging (independently-initialized
nets' weight spaces aren't aligned; averaging them wouldn't behave like either).

How permissive that merge test should be was **measured, not guessed**: a 7-point sweep found a
real cliff, not a slope — stable settling at merge-tolerance 1 through 4 (63–92 bricks), a
borderline-unstable regime at 5 (never truly settles, bottomed at 3 bricks on one run), and total
collapse into a single brick at 6 and above, confirmed at three separate values. Shipped one full
step below that cliff, at the tolerance that also happened to produce the best measured quality of
the entire sweep — not a safety/quality tradeoff, the safe choice was also the best one.

## What's here

| file | what it is |
|---|---|
| `src/native_quinary_charlm.py` | the core lossless quinary mechanism — residual-carry update, exact zero-tolerance conservation check, forward/backward through the real SIMD kernel |
| `src/quinary_simd_kernel.c` + `build_quinary_dll.bat` | the multiply-free SIMD quinary matmul kernel (AVX2) |
| `src/brick_cluster_charlm.py` | the growable cluster: the LSL-style gate, online growth, periodic consolidation |
| `src/lutgtlm_ingest.py` | a real production entry point — continuous ingestion into a persistent, growable mind (reference usage; paths are illustrative) |
| `tests/cluster_bricks_test.py` | the oracle-routing proof (content-clustered bricks beat the single net) |
| `tests/lsl_gate_routing_test.py` | the real-gate proof (fitted routing recovers oracle performance without cheating) |

These test scripts expect your own plain-text training data (a `voice_corpus.txt`-shaped file in
the working directory) — none is bundled here.

## Honest scope

The quinary level and its residual-carry update are exactly, provably lossless — proven via a
zero-tolerance conservation check (`total_in == consumed + discarded + residual`), not assumed.
The gate is a real, fitted statistical model, not a proof — it is measured to recover oracle
routing performance, not guaranteed to. Growth and consolidation are tuned against one real corpus
at one scale; the safety-margin sweep is real data, not a universal constant — re-verify against a
sweep, don't just trust a carried-forward number, if you change the corpus, scale, or feature
representation.

## License

[PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) — free
for any noncommercial purpose (personal, research, nonprofit, education, government). Commercial
use is not covered by this license — contact the author to arrange a commercial license.
