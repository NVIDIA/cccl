# Histogram selector — plan + final sweep

Evidence: `run_2026-06-15_coalesce`, `run_2026-06-13_u64`, `run_2026-06-13_lowbin_smem`.

## Selector changes
- Cache kernels (`direct_cuckoo`, `direct_single_probe`): **coalesce OFF** (1.4–1.5× faster, both entropy classes; static per-kernel const, no branch). Keep ON for `direct_nocache` + gmem gather.
- High-bin workhorse = **`direct_single_probe` coalesce-off** above the on-chip cap, all binaries (1.3–1.7× over current default).
- Drop `hybrid` + gmem gather from high-bin routing; switch multi_range 131072+ cuckoo→single-probe.
- Delete `gmem_privatized_cuckoo` / `gmem_privatized_single_probe` (never best/within-2%).
- Keep as-is: counter-width gate (`sizeof(CounterT)≤4 → static` at ≤256) and both SMEM kernels.

On-chip cap (byte-derived) = `(232448−4096)/(counter_bytes×channels)`: 4-byte → 57 088 single / 19 029 multi; 8-byte halves it. **Crossover (~57 344 single-ch) was never measured** — the sweep must.

## Final sweep (`run_perbinary_sweep.sh`, run per counter width)
- Binaries: even, range, multi_even, multi_range. Samples: I32, F64.
- Counter: 32-bit + 64-bit (u64 leg → elements to 8G; 32-bit to 2G).
- Bins: 16, 32, 64, 256, 1024, 4096, 16384, 32768, **49152, 57344**, 65536, 131072, 262144, 1048576.
- Shapes: all 14. Hit rates: pass 2 (cuckoo + single-probe).
- Columns: `main`, `default` (new selector), forced `smem_static`, `smem_dynamic`, `hybrid`, `gmem_privatized_nocache`, `direct_single_probe`, `direct_cuckoo` (each coalesce on + `__nocoal`). Exclude GPC/GPS.
- Output: per-shape speedup figs + hit-rate panels + wins table; confirms single-probe dominance and the exact crossover bin.
