# Design proposal: SMEM cache front-end for the GMEM-privatized histogram

**Status:** FULLY IMPLEMENTED (2026-06-04, B200). The ENTIRE design doc is built:
the kernel rename scheme (`DirectKernel`/`SmemPrivatized*`/`GmemPrivatized*`), the
`Combiner ∈ {NoCache, Cuckoo, SingleProbe}` axis, the hybrid↔gather merge into one
`GmemPrivatizedKernel<…, HybridSplit>`, the shared gather helper, the renamed
`algorithm` enum, and the proposed `gmem_privatized_{cuckoo,single_probe}` wired as
first-class (reachable, selectable-but-unselected) dispatch cases. Default selection is
metric-neutral; all catch2 suites pass for every algorithm; the C-parallel NVRTC
contract compiles. See "Implementation status" and "Measured results" below.
**Scope:** high-bin histogram sweep (single-channel first), bins ≤ 65 536.

## Summary

Add an **associative SMEM cache** (cuckoo / single-probe) as a write-combining
front-end to the **per-block GMEM-privatized** histogram. Cache hits combine in
SMEM; misses spill **block-scope** (`atomicAdd_block`) into the block's private GMEM
histogram; the existing grid-sync + atomic-free gather merges the private copies
into the output.

This is hot-aware on-chip combining (unlike the privatized kernel's static SMEM
tier) **with** a contention-free spill (unlike the direct kernel's device-scope
atomic to the shared output). It is the empty cell in the kernel matrix below: the
direct kernel's cache front-end crossed with the privatized kernel's backing store.

## Implementation status (what was built)

Every item in this doc is implemented in `cub/cub/device/dispatch/` (kernels +
dispatch) and `c/parallel/src/histogram.cu` (NVRTC name strings):

- **Kernel renames** (Naming scheme table): `DeviceHistogramDirectAtomicKernel →
  DeviceHistogramCacheSpillKernel`; `DeviceHistogramSmemPriv{,Dynamic,DeviceInit}Kernel
  → …SmemPrivatized{,Dynamic,DeviceInit}…`. `DeviceHistogramInitKernel` kept (an
  output-zeroing helper orthogonal to the combiner×commit taxonomy; renaming it
  would only churn the C-parallel NVRTC contract for no clarity gain).
- **Combiner axis**: `Combiner ∈ {no_cache_probe, cuckoo_cache_probe<>,
  single_probe_cache}`, all spill-agnostic (templated on a `SpillOp` ∈
  {`output_atomic_spill`, `private_block_spill`}).
- **Consolidation #1 (hybrid ↔ gather merge)**: one
  `DeviceHistogramGmemPrivatizedKernel<…, bool HybridSplit>`. `HybridSplit=false`
  (smem_split=0) is the pure GMEM-privatized gather; `HybridSplit=true`
  (smem_split>0) is the SMEM-primary + GMEM-tail hybrid. `if constexpr` selects the
  body; the hybrid branch keeps its fused primary+secondary reduce loop (the shared
  helper's two-call form measured ~5% slower there — see Measured results).
- **Anti-redundancy #2**: one `gather_privatized_slab` `__device__` helper; the
  pure-gather member and the DirectKernel private-spill Phase-4 both call it.
- **Enum rename**: `smem_priv_256` + `smem_priv_dynamic` → `smem_privatized`;
  `hybrid_single_pass` + `gmem_priv_gather` → `gmem_privatized_nocache`;
  `direct_atomic_cuckoo/_single_probe` → `direct_cuckoo/direct_single_probe`; added
  `direct_nocache`, `gmem_privatized_cuckoo`, `gmem_privatized_single_probe`.
  `dispatch_by_algorithm` recovers the merged tiers (static/dynamic; hybrid/gather)
  at runtime from the bin count / smem_split.
- **Proposal wired**: `gmem_privatized_{cuckoo,single_probe}` are first-class
  `dispatch_by_algorithm` cases (the DirectKernel cache front-ends re-pointed to
  `private_block_spill` + the gather). `select_algorithm` never returns them
  (per Decision), but they are reachable via dispatch and the `CUB_HISTO_FORCE_ALGO`
  env hook for sweeps.

**Validation**: default selection is metric-neutral (4-metric harness within 0.1% of
pre-change: 854/626/786/637); all four catch2 suites pass on the default path AND
under each forced algorithm (53 587 + 8 + 336 + 18 assertions); the C-parallel
`histogram.cu` object compiles against the renamed NVRTC strings.

## Naming scheme

Three sweep kernels, divided by **where increments land**. The on-chip combiner is a
template argument, not part of the kernel name.

- **`DirectKernel<Combiner>`** — commits each increment device-scope to the **shared
  output** (this is the honest meaning of today's "direct atomic"). No private copy.
- **`SmemPrivatizedKernel`** — whole histogram lives **on-chip**; atomic-merge to
  output. Low bins only. Retains a compile-time static-SMEM specialization for the
  fixed 256-bin tier (the hot byte-sample image path); larger sizes use dynamic SMEM.
- **`GmemPrivatizedKernel<Combiner, smem_split>`** — per-block histogram in **GMEM**,
  gather-merged. SMEM is a front-end: either a static low-bin tier (`smem_split`
  bins, no tags, no misses) **or** an associative cache (`Combiner`). Always dynamic
  SMEM.

`Combiner ∈ {NoCache, Cuckoo, SingleProbe}`. `smem_split` = number of low-index bins
held in SMEM (only meaningful for `NoCache`, where it is a static direct-mapped tier).

| Proposed enum | Proposed kernel (instantiation) | Today's enum | Today's kernel |
|---|---|---|---|
| `direct_nocache` | `DirectKernel<NoCache>` | *(none)* | `DirectAtomicKernel<no_cache_probe>` *(uncommitted ablation)* |
| `direct_cuckoo` | `DirectKernel<Cuckoo>` | `direct_atomic_cuckoo` | `DirectAtomicKernel<cuckoo_cache_probe>` |
| `direct_single_probe` | `DirectKernel<SingleProbe>` | `direct_atomic_single_probe` | `DirectAtomicKernel<single_probe_cache>` |
| `smem_privatized` | `SmemPrivatizedKernel` *(static)* | `smem_priv_256` | `SmemPrivKernel` |
| `smem_privatized` | `SmemPrivatizedKernel` *(dynamic)* | `smem_priv_dynamic` | `SmemPrivDynamicKernel` |
| `gmem_privatized_nocache` | `GmemPrivatizedKernel<NoCache, split=0>` | `gmem_priv_gather` | `GmemPrivGatherKernel` |
| `gmem_privatized_nocache` | `GmemPrivatizedKernel<NoCache, split>0>` | `hybrid_single_pass` | `HybridSinglePassKernel` |
| **`gmem_privatized_cuckoo`** | **`GmemPrivatizedKernel<Cuckoo, split=0>`** | **— proposed —** | **— proposed —** |
| **`gmem_privatized_single_probe`** | **`GmemPrivatizedKernel<SingleProbe, split=0>`** | **— proposed —** | **— proposed —** |

(Kernel names elide the `DeviceHistogram` prefix for width.) The proposal is the
bottom two rows.

## Two consolidations this naming makes obvious

1. **`hybrid` and `gmem_priv_gather` are one kernel.** Verified in
   `AccumulatePixelsHybrid` (`agent_histogram.cuh:514-607`): hybrid routes a sample
   to SMEM when `bin < split` and to the per-block private GMEM slab otherwise —
   **both via `atomicAdd_block`** — then gathers over both regions. That is exactly
   `gmem_priv_gather` (all bins in GMEM) with the low `split` bins promoted to SMEM.
   Same backing store, same commit; only the storage tier of the low bins differs.
   They collapse into `GmemPrivatizedKernel<NoCache, smem_split>`: `split=0` is the
   old gather kernel, `split>0` is hybrid.

2. **The proposed kernel reuses both halves of existing kernels.** Its combiner is
   the same `ProbeOp` the `Direct` kernels use; its backing store + gather phase are
   the same the `GmemPrivatized<NoCache>` instantiations use. Building it is wiring,
   not new machinery (see Anti-redundancy).

## The proposed algorithm

`gmem_privatized_cuckoo` / `gmem_privatized_single_probe`: take the cuckoo /
single-probe SMEM cache out of the `Direct` kernel and re-point its **miss** from
`atomicAdd(&output[bin], …)` (device-scope, shared output) to
`atomicAdd_block(&private_histogram[bin], …)` (block-scope, per-block copy), then let
the existing gather merge the copies.

Why the gap is worth filling:

- **`hybrid`'s SMEM tier is hotness-blind** — it dedicates ~49 152 slots to bins
  `[0, split)` by index. On inputs whose hot bins are **scattered across the range**,
  those slots sit on cold low-index bins while the hot bins pay the GMEM tier.
- **`direct_{cuckoo,single_probe}` is hot-aware but its miss is contended** — a
  device-scope atomic on the shared output, the expensive event the cache exists to
  avoid.
- **`gmem_priv_gather` spills contention-free but has no on-chip combiner** — every
  sample pays a GMEM round-trip, which is why tuning never selects it.

The proposed path is the missing combination: **hot-aware combining + contention-free
spill**. It should win where hybrid is weakest (sparse / scattered hot bins) and lose
where the active set is dense in `[0, split)` (hybrid's dedicated 4 B/slot mapping
holds ~12× more bins than an 8–20 B tagged slot, with no CAS/hash/flush). The net
winner per input shape is an empirical question for the sweep, not an a-priori one.

## Anti-redundancy

Two extractions remove existing duplication and make the proposal free:

1. **Spill-agnostic probe ops.** Today each probe op (`cuckoo_cache_probe`,
   `single_probe_cache`, `no_cache_probe`) hardcodes its miss as
   `atomicAdd(&output[bin], …)`. Replace that with a `Spill` functor argument:
   - `output_spill` → `atomicAdd(&output[bin], …)`        *(used by `DirectKernel`)*
   - `private_spill` → `atomicAdd_block(&priv[bin], …)`   *(used by `GmemPrivatizedKernel`)*

   The probe ops (hash, slot, CAS, hit-combine, flush) become reusable verbatim
   across both kernels. 3 combiners × 2 landings = 6 behaviors from 3 + 2 small ops.

2. **Shared gather-merge helper.** The
   `for bin: total += base[b*num_bins + bin]; out[bin] = total` reduce already exists
   twice — `GmemPrivGatherKernel` (Phase 4) and `HybridSinglePassKernel` (the
   atomic-free reduce). Extract one `__device__` helper; all `GmemPrivatized`
   instantiations call it.

The cache's dynamic-SMEM sizing (`cache_tuning::slots_floor`, occupancy-preserving
growth) is reused **as a starting point** from the direct-atomic sizer — but whether
that sizing is right for either kernel is an open question (see Cache sizing below).

## Dispatch integration

Add `algorithm::gmem_privatized_cuckoo` / `…_single_probe` and a `select_algorithm`
rule scoped to the tier below. When selected, dispatch allocates the same
`num_blocks × num_bins × sizeof(LocalCounterT)` private-histogram temp storage that
`gmem_priv_gather` uses, plus the cache's dynamic-SMEM reservation. No new sizing
logic; the combiner and `smem_split` are template arguments threaded through the
existing `GmemPrivatized` launch path.

## Scope / when selected

An associative cache can miss on **any** bin, so the private histogram must be
**full-size** (`num_blocks × num_bins`) — boundedness and hotness-blindness were the
same decision in hybrid. At ~256 blocks: 65 536 bins → ~64 MB (fine);
1 048 576 bins → ~1 GB plus a gather that re-reads it all, where the init+gather tax
dominates (the exact cost the high-bin `Direct` path was built to avoid).

Target **single-channel, bins ≤ 65 536** (hybrid's regime). Multi-channel multiplies
the private histogram by channel count — defer.

## Cache sizing: RESOLVED by Step 0 measurement (2026-06-04, B200)

Step 0 ran the no-code `CUB_HISTO_FORCE_SLOTS` sweep on the existing `Direct` kernel
(single-channel even+range, bins ∈ {262144, 1048576}, elements 64M, shapes
{concentrated:0.0, powerlaw:0.5, hash_synonym, stale_resident}, 3 repeats, CoV≈0).
Raw data: `autocuda/results/step0_direct_slot_sweep.txt`. Result — the "more SMEM,
lower occupancy" premise is **falsified for `Direct`**:

- **4096 → 8192/auto: flat** in 15/16 cells (within ±1%). The growth the sizer *does*
  allow (it stops at the occupancy-free point) buys ~nothing on skewed inputs.
- **Forcing past the occupancy-free point (→16384, where occupancy collapses):
  catastrophic 3–6× cliff** in every cell (e.g. even/1M/powerlaw 555→165 GiB/s;
  range/1M/hash_synonym 475→79). The occupancy-preserving sizer is exactly right.
- Lone exception: even/1M/stale_resident 4096→8192 = +2% (486→496), an adversarial
  floor, not a real workload — noise-adjacent and not worth a sizing change.

**Consequence for the proposal:** the *cache-sizing* lever is dead — for `Direct` and,
by the near-symmetry argued below, almost certainly for `GmemPrivatized` too. What Step
0 did **not** test is the proposal's *other, orthogonal* claim: **contention-free spill**
(miss → block-scope private copy vs `Direct`'s device-scope atomic to the shared
output). That is independent of cache size and remains open. It can only pay where the
miss stream is **both frequent and contended** — i.e. the adversarial shapes where hot
traffic provably escapes the cache (`hash_synonym`: 32 hot bins on one slot → 31
permanently-missing hot bins; `stale_resident`: working set thrashes the cache). The
revised, narrowed thesis the prototype must test: *does private-spill rescue the
adversarial-shape floors that `Direct`'s contended spill craters, without losing the
non-adversarial cells to the gather tax?*

The original open-question text is kept below for context, now answered for `Direct`.

## Cache sizing: the original open question (now answered for Direct)

The existing `direct_atomic` sizer grows the cache only while occupancy stays at the
floor occupancy ("free SMEM"), then stops. It **never tests growing past that point**
at the cost of occupancy, so the code is silent on whether a bigger cache at lower
occupancy would win. Its occupancy-preserving rule is a *geomean-over-the-input-mix,
multi-channel* bet (most inputs gain nothing from extra slots, and occupancy matters
on average) — not a proof that more SMEM never helps a *specific* skewed input.

For a skewed input whose **hot set exceeds the slot floor**, growing the cache at the
cost of occupancy plausibly helps — and this argument applies to **`Direct` and
`GmemPrivatized` alike**, not just the proposed kernel:

- More slots → more of the hot set is cached → fewer spills (identical mechanism in
  both kernels).
- Growth simultaneously *removes* misses and *lowers* occupancy — and for a kernel
  that is latency-bound **on its misses**, the freed warps were hiding misses that no
  longer exist. The lever partly self-justifies.

The two kernels differ only at the margins, and the effects nearly cancel:

| | `Direct` + cache | `GmemPrivatized` + cache |
|---|---|---|
| Occupancy cost of growth | **higher** — remaining misses are *contended* device atoms, more warp-hungry to hide | lower — misses are uncontended (block-scope into the cache-resident private copy) |
| Benefit per captured bin | **higher** — each cached bin removes a *contended* spill | lower — removes only a cheap uncontended spill |
| Gather rebate from lower occupancy | none (no gather) | **yes** — fewer blocks ⇒ smaller `num_blocks × num_bins` init + gather |

The first two rows pull opposite ways and roughly cancel, so there is **no clean
"only GmemPrivatized wants a big cache" asymmetry**. The one durable difference is the
gather rebate: lower occupancy shrinks `GmemPrivatized`'s dominant overhead, while
`Direct` gets nothing back for lost occupancy. That nudges `GmemPrivatized` toward a
larger-cache / lower-occupancy operating point — but it is a weak, bin-count-limited
effect, not a qualitative split.

Growth is net-negative for *either* kernel only when it fails to raise the hit rate:
the hot set already fits the floor (most inputs), or it is so large nothing helps
(uniform ~1M bins, cache ≈0% effective at any size). Neither is the scattered-
moderate-hot-set regime this kernel targets.

**Conclusion:** treat "grow the cache past the occupancy-free point" as an unresolved,
measurable question for both cache kernels, not as settled by the current sizer. The
prerequisite sweep below tests it directly.

## Evaluation plan

Use the sweep harness in the `hist-sweep-viz` worktree.

**Step 0 (prerequisite — no new code): does a bigger cache at lower occupancy help
the EXISTING `direct_{cuckoo,single_probe}` kernel?** This gates the whole proposal.
If forcing the cache past the occupancy-free point never helps even `Direct`, the
"more SMEM, less occupancy" premise is dead and `GmemPrivatized`+cache inherits a
weaker case. The knobs already exist — no kernel changes:
  - `CUB_HISTO_FORCE_SLOTS` to push slots past the sizer's free point;
  - the `minBlocks` launch-bound hint (or `CUB_HISTO_FORCE_DA_THREADS`) to force the
    low-occupancy operating point;
  - sweep on **skewed / scattered-hot-bin** inputs at 16 384 / 65 536 bins;
  - capture `ncu` L2-atomic-latency and cache-hit-rate counters to confirm *why* it
    moves, not just that it moves.
  Run this for `Direct` first; if it shows a win band, repeat the same forced sweep
  for `GmemPrivatized`+cache once it exists (step 2) to test the gather-rebate edge.

1. Make the probe ops spill-agnostic; add `private_spill`; wire the cache front-end +
   shared gather into the `GmemPrivatized` path.
2. Force-select via the env hook (`sweep_force_is(...)` / `CUB_HISTO_FORCE_SLOTS`); no
   tuning changes yet. Sweep the cache size × occupancy operating point for the new
   kernel as in Step 0, not just the occupancy-preserving default.
3. Sweep input shape (uniform / constant / skewed / **scattered hot bins**) ×
   bin count {16 384, 65 536} against `hybrid_single_pass` and
   `direct_{cuckoo,single_probe}`.
4. Promote into `select_algorithm` only for cells where it is the geomean winner;
   otherwise leave it selectable-but-unselected (like `gmem_priv_gather` today).

## Measured results (2026-06-04, B200, single GPU)

Measured through the **fully-wired dispatch** (the proposed algorithms are
first-class `dispatch_by_algorithm` cases with real temp-storage allocation + grid
sizing, not an env-hook bolt-on). `CUB_HISTO_FORCE_ALGO` forces any algorithm at any
high-bin cell for apples-to-apples comparison. The output-spill / incumbent paths are
metric-neutral vs pre-change (4-metric harness within 0.1%: 854/626/786/637).

**Correctness:** all four catch2 histogram suites pass on the default path AND under
each forced algorithm — `direct_{nocache,cuckoo,single_probe}`,
`gmem_privatized_{nocache,cuckoo,single_probe}` — at 53 587 + 8 + 336 + 18 assertions
each, plus the benchmark's inline verifier on every swept cell.

**Performance:** forced matrix = 4 binaries × {65 536, 262 144, 1 048 576} bins ×
{concentrated:0.0, powerlaw:0.5, hash_synonym, stale_resident} × 64 M elements,
3-sample median (data: `autocuda/results/step2_postrework_matrix.csv`).
**The proposal (`gmem_privatized_{cuckoo,single_probe}`) wins 1 of 48 cells.**

Per-binary geomean GiB/s (all bins × shapes):

| binary | default | gmem_priv_nocache | direct_cuckoo | direct_single_probe | **gmem_priv_cuckoo** | **gmem_priv_single_probe** |
|---|---|---|---|---|---|---|
| even        | 833 | 702 | 770 | 775 | **475** | **481** |
| range       | 663 | 525 | 667 | 673 | **402** | **422** |
| multi_even  | 718 | 470 | 708 | 718 | **487** | **491** |
| multi_range | 595 | 162 | 589 | 595 | **162** | **162** |

The proposal is **~1.5–4× slower in aggregate**, and the gap widens with bin count and
channel count — the `num_blocks × num_bins × channels` slab zero-init + gather
dominates, exactly the gather tax this doc predicted.

**The single win cell** is **multi_even 262 144 `powerlaw:0.5`** (the realistic
multi-hot warm set): `gmem_privatized_single_probe` = **453** vs the best incumbent
372 (`gmem_privatized_nocache`) and default 303 — **+22%**. (Wiring the proposal
through proper dispatch sharpened this vs the earlier env-hook prototype: one cell,
larger margin, instead of two marginal cells.) The thesis works exactly as argued: a
warm set overflows the cache, the residual spill is frequent AND contended, and
multi-channel amplifies the contention (3 channels racing the same output bins), so
the contention-free private spill wins.

But the win is **unexploitable**:

- It is **one shape in one cell**. On the same `(multi_even, 262 144)` cell the
  proposal *regresses* the other shapes badly — concentrated:0.0 2233→962 (−57%),
  hash_synonym 597→346 (−42%) — because the cache already absorbs their hot bins, so
  the gather tax is pure loss. At 65 536 it loses on powerlaw too (538 vs 587); at 1 M
  it collapses (225 vs 371).
- `select_algorithm` **cannot observe input shape**. Routing `(multi_even, 262 144)`
  to the proposal would also catch concentrated / hash_synonym inputs there and crater
  them. Net over the shape mix is negative.
- The win cell is a **low-throughput** cell (~300–450 GiB/s); +22% there is a small
  absolute gain on the slow tail, not on the cases that dominate the geomean.

## Decision

**Do not promote `priv_{cuckoo,single_probe}` into `select_algorithm`.** The
private-spill kernel is correct and occasionally optimal, but only on a shape the
selector can't detect, at a margin that the same routing erases elsewhere. It stays
**selectable-but-unselected** behind `CUB_HISTO_FORCE_ALGO`, exactly the status of
`gmem_priv_gather`.

**Keep the spill-agnostic refactor.** It is zero-cost (output-spill codegen
byte-identical, regression-checked), it makes the (cache × spill) matrix expressible
without kernel duplication, and it leaves a correct, tested private-spill kernel one
selector-rule away should a future arch shift the balance (e.g. cheaper grid-sync /
gather, or a shape-aware dispatch signal) — the design doc's "third corner" is now
built and measured, not hypothetical.

## Risks / open questions

- **Gather tax vs contention saved.** Below the contention regime the gather may cost
  more than the device-scope atomics it removes — the same reason `gmem_priv_gather`
  is unselected today. The cache has to tip that balance; the sweep decides.
- **Does the cache rescue the privatized path at all?** If the bottleneck was the
  *absence* of combining (not the spill), the cache may make it competitive — or just
  confirm hybrid / direct already cover the space. Either outcome is informative.
- **Static-SMEM specialization.** Keeping `SmemPrivatizedKernel`'s compile-time
  256-bin static path (the hot byte-sample image case) means `smem_privatized` is one
  enum but two codegen paths; do not let the unification force that tier onto dynamic
  SMEM.
- **Win region may be empty.** If dense-`[0, split)` inputs dominate the benchmark
  mix, hybrid wins every measured cell and this stays unselected. The residual value
  is then closing the (combiner × landing) matrix and giving the privatized-cache
  cell a measured verdict.
