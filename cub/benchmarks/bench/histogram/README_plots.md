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

`hybrid_single_pass` is single-channel-only
and omitted from the multi-channel folders.

```
python histogram_algo_perf.py --results sweep_results.json --outdir algo_perf_figs
```

The sweep JSON is produced by a scratch driver that forces each candidate via a
`CUB_HISTO_FORCE_ALGO` env hook (sweep-only scaffolding, not shipped) and validates
forced==launched from `CUB_HISTO_LOG_LAUNCH` output. The two scripts here only
consume the resulting JSON.

Both scripts need `numpy` + `matplotlib`.
