# Composed Tile worker-width benchmark

`bench.py` measures integer scan, reduction, and scan followed by reduction,
with pointwise work before and after the provider call. Every kernel performs
its full load/compute/store sequence. Results are checked against an independent
int64 reference, including an int32 range check, before timing.

Use `cuda-coop-tile` with the matching `cuda-tile` and `cuda-tile-preview`
1.8.0 packages and CUDA 13.5. Building the provider requires the separately
supplied SDK described in the
[installation guide](../../../docs/tile/python/installation.rst).
The benchmark also imports PyTorch.

From the repository root:

```bash
python python/cuda_coop_tile/benchmarks/bench.py --threads 128 256 \
  --elements 33554432 --output results.json
```

The script uses the same input/output allocations for every case. The example
has a 256 MiB combined working set; choose `--elements` deliberately relative to
the tested GPU's cache capacity. By default, CUDA events time graphs of 32
kernel launches, after 20 warmup replays, with 31 samples in each of three
randomized rounds.
JSON retains every sample, workload size, reference bounds, and median/quartiles.
Compilation, allocation, correctness checks, and Python dispatch are excluded.

The default extents are 256, 1024, and 4096; `--extents` changes them. Every
extent must divide `--elements`. Only worker widths that divide each extent
are measured. `--scenarios` selects `scan_affine`, `reduce_affine`, or
`scan_reduce`; all three run by default.

Use worker widths supported by the installed frontend and the selected
primitive. This benchmark does not change compiler validation or measure
automatic provider ranking.
Results apply to the chosen integer workloads, allocation reuse, frontend,
provider, compiler, and GPU; they do not
establish a universal provider ranking.
