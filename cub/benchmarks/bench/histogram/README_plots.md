# Histogram benchmark input-shape & algorithm plots

Two plotting scripts live here, next to the benchmark sources and the input-shape
generators they visualize.

## `histogram_input_design.py` — source of truth (shared module)
A bit-exact host-side Python mirror of the input-shape generators in
`histogram_inputs.cuh`. For a given `(shape, n, num_bins, seed)` it produces the
same per-element bin indices the benchmark produces on device. Both plotting
scripts import it; keep it in sync if `histogram_inputs.cuh` changes (run its
`__main__` self-test to spot-check). It only needs `numpy`.

## 1. `histogram_input_characterization.py` — what each InputShape looks like
One figure per InputShape, three panels:
- **value distribution** — count vs bin index on a **log y-axis**, one stem per
  occupied bin. (A linear count-vs-index plot makes these look empty: a single
  hot bin is a sub-pixel spike and a decaying tail is crushed flat under it. Log-y
  stems show the spike(s) and the floor.)
- **rank-frequency** — sorted count vs rank, log-log. Scale-free: a power law is a
  straight line, a single hot bin one point over an empty floor, `hash_synonym` a
  few high points over a flat floor. Reveals the shape regardless of bin count.
- **position of values** — bin index vs position in the input sequence. Ordering
  shapes (`temporal_phases`, `stale_resident`, `sawtooth`) show their structure here;
  `concentrated:1.0` (uniform) is a random scatter.

Each shape is drawn at its **natural bin count** (`CHAR_BINS_BY_SHAPE`), not one
global value: the i.i.d. distribution shapes put their hot bin at `seed % num_bins`
(= 42), so at a few hundred bins it sits visibly off zero (at 16384 bins it would
read as "pinned to 0"); the cache-adversarial shapes need bins > 4096 to show their
structure (hash_synonym's 4096-spaced synonyms; stale_resident's 4096-bin prefix).

```
python histogram_input_characterization.py --outdir histogram_input_figs
# knobs: --elements --bins (override per-shape) --seed --shapes <subset>
```

## 2. `histogram_algo_perf.py` — algorithm performance vs #bins
Reads a per-cell sweep JSON (each high-bin algorithm forced across
`SampleT × Elements × Bins × InputShape`) and writes one image per InputShape into
`<even|range>_<single|multi>_<I32|F64>/`. Each image has the shape's
characterization (top row, drawn by the shared functions above so it matches the
characterization figures exactly) and, below, GiB/s vs #bins — one connect-the-dots
line per algorithm, per element count.

> Note: if you change an InputShape's generator in `histogram_inputs.cuh` (e.g. the
> `concentrated:1.0` uniform endpoint was changed from a sequential ramp to an
> exact-count Feistel shuffle), the perf JSON's column for that shape is stale until
> you re-run the sweep. The characterization top row reflects the current generator;
> the perf grid reflects whatever the sweep last measured.

A forced algorithm that does not apply at a given `(transform, channels)` simply
has no measured points and is dropped from that panel.

```
python histogram_algo_perf.py --results sweep_results.json \
    [--hitrate hitrate_results.json] --outdir algo_perf_figs
```

## 3. `histogram_algo_sweep.py` — produce the perf JSON (incl. upstream `main`)
Drives the sweep the plot script consumes. For every
`(binary, SampleT, Elements, Bins, InputShape)` cell it forces each high-bin
algorithm via `CUB_HISTO_FORCE_ALGO` (the six gmem-privatized / direct-atomic
variants) and also records the selector's own pick as `default`. Forcing is only
honored above the high-bin threshold (bins > 4096); at/below it every forced algo
falls back to `smem_privatized`, so low-bin cells are recorded once under
`default`. `CUB_HISTO_DEBUG_SLOTS` is set so a direct-atomic kernel that actually
ran prints a tell — recorded per cell (`dr=1`) to catch a silent fallback.

Pass `--main-bin-dir` pointing at benchmark binaries built from upstream `main`
to add a `main` column = main's default dispatch (main has no force hook).

**For an all-shapes comparison, build the main baseline with THIS branch's
input-shape generators.** The branch's two later input-shape commits
(`0cf4594ba6` sawtooth/random-order/drop-capacity_cliff, `286a78e248` redefine
concentrated/stale_resident) are **bench-only** — they touch no dispatch/kernel
code — so overlaying this branch's `histogram_inputs.cuh` / `*.cu` onto a stock-`main`
checkout yields *main's dispatch with identical generators*. Then every shape is
apples-to-apples and only the dispatch differs (the comparison we want). Setup:

```
git worktree add ../main-baseline main
cp cub/benchmarks/bench/histogram/{even,range,histogram_inputs}.cuh? \
   ../main-baseline/cub/benchmarks/bench/histogram/         # + multi/{even,range}.cu
# configure+build the 4 cub.bench.histogram.*.base targets in the main worktree
```

With that, the sweep compares **all** swept shapes by default (no restriction).

If instead you point `--main-bin-dir` at UNMODIFIED upstream-main binaries (whose
generators differ for `concentrated:*` / `stale_resident`, and which lack
`sawtooth`), restrict the comparison to the generator-identical subset:
`--main-comparable-shapes powerlaw:0.5 zipf:1.0 hash_synonym temporal_phases strided_sweep`.

```
python histogram_algo_sweep.py \
    --branch-bin-dir build/autocuda/cub-benchmark/bin \
    --main-bin-dir   ../main-baseline/build/cub-benchmark/bin \
    --samples I32 F64 --out sweep_results.json
# knobs: --bins --elements --samples --shapes --binaries --repeats --timeout
#        --main-comparable-shapes (only for unmodified-main baselines)
```

This is the shipped, reproducible successor to the older scratch force-hook driver
(it validated forced==launched from the `CUB_HISTO_DEBUG_SLOTS` tell). The two
plotting scripts above only consume the resulting JSON.

### Optional: SMEM-cache hit-rate panels (two-pass sweep)
The cached high-bin kernels (`cuckoo`, `single_probe`) can be built with
`-DCUB_HISTO_TRACK_HITRATE=1` to count, per launch, the contributions ABSORBED in
the SMEM cache (a block-scope `atomicAdd` — a hit) vs SPILLED to a GMEM atomic (a
miss), weighted by the warp-coalesced contribution so the rate is over input
elements. The instrumentation is **zero-cost when the macro is off** (no extra
`apply()` args, no accumulators — SASS-identical to the uninstrumented,
register-pinned kernel); the host reads the grid-wide totals back via
`cudaMemcpyFromSymbol` and prints `[hitrate] ...` under `CUB_HISTO_LOG_HITRATE=1`.

Because the readback adds overhead, the parameter space is swept **twice**: once
with the normal binaries for performance (`sweep_results.json`), and once with the
`*.hitrate` binaries for hit rate (`hitrate_results.json`). The hit-rate pass uses
NVBench `--profile` (one measured launch per cell — hit/miss counts are
deterministic) and a single sample type (hit rate is sample-type-independent). When
`--hitrate` is supplied, each per-shape image gains a bottom row: **cuckoo hit-rate
vs #bins** and **single-probe hit-rate vs #bins**, each with **#elements as series**.

Both scripts need `numpy` + `matplotlib`.
