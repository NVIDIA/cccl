# Histogram work: unification, sweeps, visualization — status quo & merge plan

Scope: consolidate the recently active histogram branches into one
`autocuda/best/<date-time>` branch, and split out the non-optimization changes
(correctness, testing, visualization tooling) for backport to `main`.

---

## 1. Branch lineage

All four "feature" branches share upstream `main` (`b22266b835`, which already
carries the *first* input-shape rework + the `ScaleTransform` overflow fix).

```
main (b22266b835)
│
├─ histogram-simplify-2026-06-02   (145 ahead, 0 behind)
│     └── strict ANCESTOR of cached-privatized-spill — nothing unique, fully subsumed.
│
└─ 6a44b33f58  "merge cuckoo + single-probe into one direct-atomic kernel"   ← divergence point
   │
   ├─ cached-privatized-spill        (154 ahead of main; +16 past 6a44b33f58)   ← MAIN LINE
   │     C-parallel cooperative high-bin path (NVRTC + cuLaunchCooperativeKernel),
   │     device-init kernel removal, dead-param cleanup, kernel/enum renames,
   │     the algo-sweep + plot tooling, and the upstream-`main` baseline harness.
   │
   └─ histogram-smem-capacity-exploration   (142 ahead of main; +4 past 6a44b33f58)
         4 unique commits:
           790f09e39d  cache HIT-RATE instrumentation (+106 lines in kernel_histogram.cuh)
                       + SMEM capacity-vs-occupancy exploration (+314 in dispatch_histogram.cuh)
           0caf8807e7  Lever-A figures overlay all algorithms
           991a84eeff  Lever-A figure recolor
           9f1f97b85f  Lever-A cache-size findings report
```

Relationships verified with `git merge-base`:
- `histogram-simplify` ⊂ `cached-privatized-spill` (ancestor; ignore it — promoting the
  child carries it).
- `cached-privatized-spill` and `histogram-smem-capacity-exploration` **diverge** at
  `6a44b33f58`: 16 vs 4 commits respectively.

### Other branches (NOT in scope — context)
- `backport/histogram-input-shapes` (== `main`): the already-landed first input-shape rework.
- `histogram-input-shapes` (130 ahead / 4 behind main): older input-shape line, superseded.
- `bench-correctness-checks`, `hist-bench-hardening*`, `bench-correctness-test-*`: late-May
  bench-hardening experiments, mostly behind main; treat as superseded unless a specific
  fix is missing from main (audit during backport, §6).
- `autocuda/optimize/*/worker-*`, `experiment/*`, `tmp/*`: trial/scratch worker branches — noise.

### Merge surface (cached-privatized-spill ↔ smem-capacity-exploration)
`git diff --stat` across the divergence:
- **Zero-conflict (new files):** everything under `autocuda/smem_capacity_explore/` +
  `autocuda/sweep_hitrate.py` (1834 insertions, all new paths).
- **Real overlap (2 shared headers):**
  - `kernel_histogram.cuh` — `+106` lines: the hit-rate instrumentation. Written against the
    **pre-rework** probe ops, so it must be **re-applied by hand** onto the current
    `cuckoo_cache_probe` / `single_probe_cache::apply(... cache_mask, cache_slot_log2)`
    signatures (it will NOT cherry-pick clean). Must stay SASS-neutral when the macro is off.
  - `dispatch_histogram.cuh` — `+314` lines: `CUB_HISTO_FORCE_SLOTS` cap-override plumbing +
    capacity-sweep hooks. Partially already present on the main line (force-slots exists);
    reconcile, don't double-apply.

---

## 2. Sweep tooling (status quo)

### 2a. Shipped / current (on `cached-privatized-spill`, in the bench dir)
| script | role | matrix |
|---|---|---|
| `histogram_algo_sweep.py` | **primary** perf sweep driver. Forces each high-bin algo via `CUB_HISTO_FORCE_ALGO` + records selector `default`; optional upstream-`main` column via `--main-bin-dir`. Validates forced==ran via `CUB_HISTO_DEBUG_SLOTS` (`dr` flag). | binaries {even,range,multi_even,multi_range} × algos {default + 6 forced + main} × bins {256,2048,8192,32768,65536,262144,1048576} × elements {16M,256M} × SampleT {I32,F64} × 9 shapes; 3-sample median. `HIGH_BIN_THRESHOLD=4096` (below it, forcing is a no-op → default only). |
| `histogram_input_design.py` | bit-exact host mirror of `histogram_inputs.cuh` generators; imported by the plot/char scripts. | n/a (generator) |

### 2b. autocuda/ scratch drivers (on `cached-privatized-spill`, older)
`step0_sweep.py` (forced-slots Step-0 gate), `step2_forced_matrix.py` (focused
forced matrix → `step2_forced_matrix.csv`), `step2_analyze.py`,
`run_histogram_benchmark.py`. Superseded by `histogram_algo_sweep.py` for the perf
matrix; keep for provenance, do not ship.

### 2c. Capacity / hit-rate sweeps (on `histogram-smem-capacity-exploration`)
| script | role | matrix |
|---|---|---|
| `autocuda/sweep_hitrate.py` | **pass-2** hit-rate sweep using `-DCUB_HISTO_TRACK_HITRATE=1` binaries; parses `[hitrate] … hits/misses/rate` per cached launch. Single SampleT (rate is sample-type-independent). → `hitrate_results.json` | cuckoo + single_probe × bins × elements × shapes |
| `smem_capacity_explore/sweep_cache_slots.py` | **Lever A**: sweep per-block cache SLOT count (`CUB_HISTO_FORCE_SLOTS`) past the auto-sizer. | algos {cuckoo,single_probe} × slot counts × bins × elements |
| `smem_capacity_explore/sweep_priv_cap.py` | **Lever B**: larger privatized-SMEM cap variant binaries (cap16384…49152) vs the high-bin path they'd replace. | cap variants × bins × elements |
| `smem_capacity_explore/build_hitrate_variants.sh` | surgical relink of the 4 benches with `-DCUB_HISTO_TRACK_HITRATE=1` → `*.hitrate` binaries. |
| `smem_capacity_explore/build_cap_variants.sh` | build `*.cap<N>` privatized-cap variant binaries. |

### 2d. Force/debug env hooks (in `dispatch_histogram.cuh` / `kernel_histogram.cuh`)
`CUB_HISTO_FORCE_ALGO` (pin algorithm; high-bin only), `CUB_HISTO_FORCE_SLOTS` (pin
cache slots), `CUB_HISTO_DEBUG_SLOTS` (emit chosen slots / "a direct kernel ran"
tell), `CUB_HISTO_CACHE_HASH_MODE`, `CUB_HISTO_SINGLE_PROBE_WAYS`. Hit-rate macros
`CUB_HISTO_TRACK_HITRATE` / `CUB_HISTO_LOG_HITRATE` are **documented in README_plots.md
but NOT in the current kernels** — they live only on `histogram-smem-capacity-exploration`
(this is the gap behind the missing hit-rate panels; see §5/§6).

### Sweep matrices in use (summary)
- **Main perf matrix (current):** bins {256 → 1M, 7 pts}, elements {16M, 256M}, SampleT {I32, F64}, 9 shapes, 4 binaries, 8 algo columns (incl. `main`).
- **Old richer matrix (the figures the operator remembers):** bins {32768 → 1M, 6 pts}, elements {1M,16M,64M,256M,1G,2G — 6 pts}, 1 SampleT, 5 algos (old taxonomy), **+ hit-rate row**.
- **Capacity matrix (Lever A/B):** slot counts / cap variants × high-bins × elements.

---

## 3. Visualization tooling (status quo)

| script | input | output structure |
|---|---|---|
| `histogram_algo_perf.py` | `--results <perf>.json` [`--hitrate <hr>.json`] | `algo_perf_figs/<even\|range>_<single\|multi>_<I32\|F64>/<shape>.png`. Per image: **top row** = input characterization (distribution + position-in-sequence + algorithm legend); **middle** = GiB/s-vs-#bins panels (one per element count); **bottom row** (only with `--hitrate`) = cuckoo + single-probe cache hit-rate vs #bins (series = #elements). |
| `histogram_input_characterization.py` | generators (imports `histogram_input_design`) | `histogram_input_figs/<shape>.png`, 3 panels: value distribution (log-y stems), rank-frequency (log-log), position-of-values. Also exports the shared `draw_distribution/draw_sequence/char_input/SHAPES/SHAPE_BLURB` used by the perf plotter's top row. |
| `smem_capacity_explore/plot_capacity_figs.py` | capacity sweep JSON | same folder/ә layout as `algo_perf_figs`, but each line is a **capacity setting** (Lever A slots / Lever B cap) with the status quo as a thick reference line. |

### Current `algo_perf_figs` style (after this session's fixes)
- perf panels: **log y-axis** (so a 30× slower baseline stays on-figure and the
  vertical gap = speedup).
- reference series drawn on top: `default` (thick solid black), `main` (thick
  bright-red dash-dot, X markers) + shaded speedup band between them.
- 6-algo taxonomy: `gmem_privatized_{nocache,cuckoo,single_probe}`,
  `direct_{cuckoo,single_probe,nocache}`.

### Visualized-result structure (what a reader sees)
- **Per (transform × channels × SampleT × InputShape)** → one PNG.
- Within a PNG: characterization (what the input looks like) on top; throughput vs
  bin-count for every algorithm + the two baselines in the middle; optional cache
  hit-rate at the bottom.
- **Characterization** PNGs are standalone, one per shape, at each shape's *natural*
  bin count (`CHAR_BINS_BY_SHAPE`: 256 for i.i.d. shapes so the hot bin at seed%bins
  is visible; 8192–16384 for cache-adversarial shapes whose structure is defined
  relative to the 4096-slot cache).

### Input-shape catalogue (13 characterized; 9 swept for perf)
`concentrated:{1.0,0.75,0.5,0.25,0.0}` (entropy ramp), `powerlaw:{0.75,0.5,0.25}`,
`zipf`, `temporal_phases`, `stale_resident`, `hash_synonym`, `sawtooth`.
(`capacity_cliff` was dropped on the branch; `strided_sweep` swept but minor.)

---

## 4. Known divergences to reconcile when merging

1. **Algorithm taxonomy.** Old figures/JSON use `direct_atomic_cuckoo`,
   `gmem_priv_gather`, `hybrid_single_pass`; current code uses
   `direct_cuckoo` / `gmem_privatized_nocache` / (hybrid merged into
   `gmem_privatized`, `HybridSplit=true`, no longer a forceable enum). Old sweep
   JSONs are stale; re-sweep rather than translate.
2. **Hit-rate instrumentation** exists only on `histogram-smem-capacity-exploration`,
   against pre-rework probe ops → must be re-applied to current `apply()` signature.
3. **Generator drift vs `main`.** The two later bench-only commits (`0cf4594ba6`,
   `286a78e248`) redefined `concentrated:*` / `stale_resident`, added `sawtooth`,
   dropped `capacity_cliff`. For an all-shapes `main` comparison, overlay this
   branch's generators on a stock-`main` build (they touch no dispatch/kernel code).

---

## 5. Plan A — produce `autocuda/best/<date-time>`

Goal: one branch = all optimization + tooling + the hit-rate capability, validated.

**Step 0 — name & base.** Branch `autocuda/best/<YYYY-MM-DD-HH-MM-SS>` (today) from
`cached-privatized-spill` (the superset main line; carries `histogram-simplify`).

**Step 1 — fold in the capacity/hit-rate tooling (zero-conflict files).**
`git checkout histogram-smem-capacity-exploration -- autocuda/smem_capacity_explore/
autocuda/sweep_hitrate.py` — pure new paths, no conflict.

**Step 2 — re-apply hit-rate instrumentation to current kernels.** Port the
`CUB_HISTO_TRACK_HITRATE` accumulators/readback from `790f09e39d` onto the current
`cuckoo_cache_probe` / `single_probe_cache` `apply()` (new `cache_mask,
cache_slot_log2` args + merged spill ops). **Acceptance: SASS-identical when the
macro is OFF** (verify with `cuobjdump` diff vs pre-port, per the
[chevron/SASS-neutral discipline]).

**Step 3 — reconcile `dispatch_histogram.cuh` capacity hooks.** `CUB_HISTO_FORCE_SLOTS`
already exists on the main line; take only the capacity-sweep additions not already
present. No double-apply.

**Step 4 — validate (the full gate).**
- catch2 `cub.test.device.histogram.lid_0` (expect 53587 assertions / 38 cases).
- c-parallel `cccl.c.parallel.test.histogram` (expect 302 assertions / 10 cases).
- Rebuild ALL 4 bench binaries; high-bin abort-check across tiers.
- Perf-neutrality spot check vs `cached-privatized-spill` (the macro-off SASS proof
  in Step 2 makes this a formality, but measure 65536+1M anyway).

**Step 5 — regenerate the canonical figures** (after the running all-shapes sweep
lands): perf JSON + the hit-rate pass → `algo_perf_figs/` with all 9 shapes carrying
the `main` baseline AND the hit-rate bottom row restored. Mirror to a flat path for
`scp`.

**Step 6 — push** `autocuda/best/<date-time>` (only when the operator approves).

---

## 6. Plan B — backport non-optimization changes to `main`

Principle: `main` should get **correctness, testing, and visualization/benchmark
tooling** but NOT the experimental optimization kernels (those ship via a separate,
reviewed PR). Triage the 154 commits `main..cached-privatized-spill` into:

**B1 — Correctness (backport):** e.g. `ScaleTransform` overflow already on main;
audit `bench-correctness-checks` / `hist-bench-hardening` for any fatal-on-bad-result
or row-stride-overflow-skip fix not yet on main. Cherry-pick the genuinely-corrective
ones.

**B2 — Testing (backport):** the new c-parallel high-bin cooperative test
(`test_histogram.cpp` `[cooperative]` case) — but it depends on the cooperative
dispatch, so it backports **only with** that dispatch; HOLD for the optimization PR,
do not backport to bare `main`. Backport only test changes that stand alone on main.

**B3 — Visualization / bench tooling (backport — the safe, high-value set):**
`histogram_input_characterization.py`, `histogram_input_design.py`,
`histogram_algo_perf.py`, `histogram_algo_sweep.py`, `README_plots.md`, and the
input-shape generator state in `histogram_inputs.cuh` + `even/range/multi*.cu` axis
lists (bench-only; already partially on main via the first rework). These touch no
dispatch/kernel code and make `main`'s own histogram benchmarks richer.

**B4 — Optimization (DO NOT backport to main):** the cooperative path, cache
front-ends, kernel/enum renames, device-init removal, C-parallel JIT changes. These
are the "best" branch's reason to exist; they go through normal review as their own
PR off `main`, not a backport.

**Mechanics:** create `backport/hist-viz-tooling` off `main`, cherry-pick the B3 (and
vetted B1) commits — many are bench-dir-only so they pick cleanly. Keep `autocuda/`
scratch data out of the backport (it is tracked but is experiment provenance, not a
`main` artifact). Open as a focused PR.

**Open question for review:** confirm whether `autocuda/` (currently tracked) should
be in the `best` branch at all, or `.gitignore`d — it is experiment data, not library
code. Recommendation: keep on `best` for provenance, exclude from any `main` PR.

---

## 7. Decisions needed from reviewer
1. Scope of `best`: confirm = `cached-privatized-spill` ∪ `histogram-smem-capacity-exploration`
   (incl. re-ported hit-rate). 
2. Should the regenerated figures use the **richer 6-element matrix** (matching the
   old figures) or the current 2-element matrix? (6× is ~3× the sweep time.)
3. Backport target: a single `backport/hist-viz-tooling` PR to `main` (B3 + vetted
   B1), holding B2/B4 for the optimization PR — OK?
4. `autocuda/` tracked-vs-ignored on `best`.
