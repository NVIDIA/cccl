# Design proposal: SMEM write-combining cache on the privatized-gather path

**Status:** proposal / unbuilt
**Scope:** high-bin histogram sweep (the tier where `hybrid_single_pass` and the
direct-atomic caches compete), single-channel first.

## Summary

Put the **adaptive SMEM hash cache** (cuckoo / single-probe) in front of the
**per-block privatized GMEM histogram**: cache hits combine in SMEM, cache
**misses** spill **block-scope** (`atomicAdd_block`) into the block's private GMEM
slab, then the existing grid-sync + atomic-free gather sums the slabs into the
output.

The cache is hot-aware (unlike hybrid's static bin-range split) **and** its spill is
contention-free (unlike the direct-atomic cache's device-scope atomic to the shared
output).

## Naming: this is NOT a direct-atomic variant

The cache and the **spill target** are two independent axes that the current code
conflates. "Direct atomic" is the name of one spill choice — a **device-scope
`atomicAdd` to the shared output** — so a privatized-spill path must not be called
direct-atomic; the name would assert the opposite of what it does.

| Axis | Values |
|---|---|
| **On-chip combiner** | none · static bin-range (hybrid) · **adaptive hash cache** (cuckoo / single-probe) |
| **Spill target** | **device-scope → shared output** (= "direct atomic") · **block-scope → per-block private slab → gather** (= "privatized") |

The spill axis names the kernel **family**. So:

- `direct_atomic_*` = cache → **device-scope** spill to output. Name stays honest; **untouched** by this proposal.
- `gmem_priv_gather` = **block-scope** spill to private slab → gather. **This is the home** of the proposed path.

The proposal is therefore: **add the SMEM cache as an optional front-end to
`gmem_priv_gather`** — not a new spill mode on the direct-atomic kernel. The shared,
reused code is the **cache component**, which crosses both families.

## Motivation

| Path | On-chip combiner | Spill | Footprint |
|---|---|---|---|
| `hybrid_single_pass` | **static** bin-range `[0,split)` → SMEM | block-scope → GMEM tail → gather | bounded (`num_blocks × secondary`) |
| `direct_atomic_{cuckoo,single_probe}` | **adaptive** hash cache | **device-scope** → shared output (contended) | none |
| `gmem_priv_gather` *(today)* | **none** | block-scope → private slab → gather | full (`num_blocks × num_bins`) |
| **proposed** = `gmem_priv_gather` **+ cache** | **adaptive** hash cache | block-scope → private slab → gather | full (`num_blocks × num_bins`) |

Hybrid's SMEM tier is hotness-blind: it dedicates 49 152 slots to bins `[0,split)`
by index. On inputs whose hot bins are **scattered across the range**, those slots
sit on cold low-index bins while the hot bins pay GMEM-tier bandwidth. The
direct-atomic cache *is* hot-aware, but its miss is a contended device-scope atomic
on the shared output. `gmem_priv_gather` spills contention-free but has **no**
on-chip combiner at all, so every sample pays a GMEM round-trip — which is why
tuning currently never selects it.

The proposed path is the missing combination: hot-aware on-chip combining **and**
contention-free spill. It is literally `gmem_priv_gather` with the direct-atomic
kernel's cache bolted onto its block-scope accumulation.

## Anti-redundancy: extract the cache as a shared component

Today the SMEM cache is welded into `DeviceHistogramDirectAtomicKernel`, and each
probe op (`cuckoo_cache_probe`, `single_probe_cache`, `no_cache_probe`) **hardcodes**
its miss as `atomicAdd(&output[bin], …)` — the device-scope-to-output spill.

Factor two things out:

1. **A spill functor**, so the probe op stops hardcoding the destination:
   - `output_atomic_spill` → `atomicAdd(&output[bin], …)`        *(direct-atomic; current behavior)*
   - `private_block_spill` → `atomicAdd_block(&slab[bin], …)`    *(privatized; new)*

   The probe ops (hash, slot, CAS, hit-combine, flush) become **spill-agnostic** and
   are reused verbatim by both families. 3 probes × 2 spills = 6 behaviors from
   3 + 2 small ops, no kernel duplication.

2. **The gather-merge `__device__` helper.** The
   `for bin: total += base[b*num_bins + bin]; out[bin]=total` loop already lives in
   `DeviceHistogramGmemPrivGatherKernel` (Phase 4) and in
   `DeviceHistogramHybridSinglePassKernel` (the atomic-free reduce). Extract one
   helper and call it from both — this removes **existing** duplication, and the
   proposed path reuses it for free.

With those two extractions, the proposed path is **not a new kernel**. It is
`gmem_priv_gather` instantiated with `{cuckoo | single_probe}` instead of the
implicit `no_cache`, sharing the cache component with the direct-atomic kernel and
the gather helper with hybrid.

### Where the cache lives so it serves both kernels

The cache front-end (probe + slot storage + init + flush) is the unit shared across
families. Two realistic structures:

- **(recommended) shared cache component, two kernels keep their own sweep.** The
  direct-atomic standalone sweep (vectorized loads, warp-coalesce, MRU bracket
  cache, software pipeline) and the `gmem_priv_gather` sweep stay separate; both
  `#include` the same cache component, parameterized by their spill functor.
  Minimal blast radius, both kernel names stay honest.
- **(more aggressive) one spill-neutral sweep kernel** parameterized by
  `<ProbeOp, SpillOp>`, with `direct_atomic` and `privatized` as the two spill
  policies. Maximally unifying — it would let `gmem_priv_gather` reuse the *good*
  (direct-atomic) sweep engine rather than its AgentHistogram path — but it renames
  the kernel and reworks the enum vocabulary right after the team consolidated it
  (`6a44b33f58`). Defer unless the sweep bodies prove near-identical.

Start with the recommended structure; promote to the aggressive one only if a sweep
shows `gmem_priv_gather`'s sweep is the bottleneck and the engines converge.

## Dispatch integration

Add `algorithm::gmem_priv_gather_cuckoo` / `…_single_probe` (or a `cache_mode`
sub-field on `gmem_priv_gather`, mirroring `direct_atomic_cache_mode`). These select
`gmem_priv_gather`'s existing temp-storage allocation (`num_blocks × num_bins ×
sizeof(CounterT)`) plus the cache's dynamic-SMEM sizing
(`cache_tuning::slots_floor` / occupancy-preserving growth — reused unchanged from
the direct-atomic sizer). No new sizing logic.

`select_algorithm` gains a rule, scoped to the tier in the next section, that picks
the cached-gather path over hybrid / direct-atomic where it is the geomean winner.

## Scope / when selected

Footprint bounds applicability: an adaptive cache can miss on **any** bin, so the
privatized slab must be **full-size** (`num_blocks × num_bins`) — boundedness and
hotness-blindness were the *same* decision in hybrid. At ~256 blocks:

- 65 536 bins → ~64 MB slab: fine.
- 1 048 576 bins → ~1 GB slab + a gather that re-reads it all: the init+gather tax
  dominates (the exact cost the high-bin direct-atomic path was built to avoid).

So target **single-channel, bins ≤ 65 536** (hybrid's regime) and below.
Multi-channel multiplies the slab by channel count — defer.

## Evaluation plan

Use the existing sweep harness in the `hist-sweep-viz` worktree:

1. Add `private_block_spill`, make the probe ops spill-agnostic, and add the cache
   front-end + gather to the `gmem_priv_gather` path.
2. Force-select via the env hook (`sweep_force_is(...)` / `CUB_HISTO_FORCE_SLOTS`) —
   no tuning changes yet.
3. Sweep the input-shape axis (uniform / constant / skewed / **scattered hot bins**)
   × bin count {16 384, 65 536} against `hybrid_single_pass` and
   `direct_atomic_{cuckoo,single_probe}`.
4. Promote into `select_algorithm` only for cells where it is the geomean winner;
   otherwise keep it selectable-but-unselected.

## Risks / open questions

- **Gather tax vs contention saved.** Below the contention regime the gather may
  cost more than the device-scope atomics it removes; the sweep decides. This is the
  same reason `gmem_priv_gather` is unselected today — the cache has to tip it.
- **Does the cache rescue `gmem_priv_gather` at all?** If the privatized spill was
  never the bottleneck (the *absence* of on-chip combining was), adding a cache may
  make it competitive — or may just confirm hybrid/direct-atomic already cover the
  space. Acceptable either way: worst case we've A/B-tested the cache's value
  against a contention-free spill.
- **Slab zeroing** adds a Phase-1 cost proportional to `num_blocks × num_bins`,
  partly hidden behind the cooperative launch but not free.
- **Win region exists?** If dense-`[0,split)` inputs dominate the benchmark mix,
  hybrid's dedicated 4 B/slot mapping (holds ~12× more bins than an 8–20 B tagged
  slot, no CAS/hash/flush) wins every cell and this stays unselected. The value then
  is closing the (combiner × spill) matrix and giving `gmem_priv_gather` a reason to
  exist.
