# CI bench request flow

Bench comparisons run on dedicated GPU runners via `.github/workflows/bench.yml`, dispatched automatically from `ci/bench.yaml`.

## Steps

1. Edit `ci/bench.yaml`. Add regex filters under `benchmarks.filters.cub` and/or `benchmarks.filters.python`. Uncomment at least one GPU under `benchmarks.gpus`.

   ```yaml
   benchmarks:
     filters:
       cub:
         - '^cub\.bench\.reduce\.(sum|min)\.'
       python:
         - 'compute/reduce/sum\.py'
     gpus:
       - "rtxa6000"   # sm_86, 48 GB
   ```

2. Append `[bench-only]` to commit messages while iterating. This suppresses all non-benchmark CI jobs.

   ```
   [bench-only] tune reduce block size
   ```

3. Push. CI dispatches a bench job per GPU listed. The job checks out both `base_ref` (default `origin/main`) and `test_ref` (default `HEAD`) via `ci/bench/compare_git_refs.sh`, builds CUB benchmarks in Release mode with sccache, runs targets matching the filters, and compares using `nvbench-compare`.

4. Inspect artifacts: the workflow uploads `bench-artifacts/` with per-target JSON, markdown reports, and a `summary.md`. Job step summaries show collapsed comparison tables.

5. Before final merge, reset `ci/bench.yaml` to match `ci/bench.template.yaml`. Both files must be identical or the branch-protection check fails.

## GPU pool

| Name          | SM      | VRAM    |
|---------------|---------|---------|
| `t4`          | sm_75   | 16 GB   |
| `rtx2080`     | sm_75   | 8 GB    |
| `rtxa6000`    | sm_86   | 48 GB   |
| `l4`          | sm_89   | 24 GB   |
| `rtx4090`     | sm_89   | 24 GB   |
| `h100`        | sm_90   | 80 GB   |
| `rtxpro6000`  | sm_120  | —       |

GPU runners are shared. Be intentional — prefer one representative GPU unless architecture-specific behavior is under investigation.

## Filter syntax

CUB filters are regexes matched against ninja target names (`cub.bench.<algo>.<variant>.base`). Examples:

```yaml
- '^cub\.bench\.copy\.memcpy\.base$'     # exact target
- '^cub\.bench\.reduce\.(sum|min)\.'      # all reduce sum/min variants
```

Python filters are regexes matched against relative paths under `python/cuda_cccl/benchmarks/`. Examples:

```yaml
- 'compute/reduce/sum\.py'
- 'compute/transform/.*\.py'
```

## Advanced options

```yaml
benchmarks:
  base_ref: "origin/main"       # default; any ref or SHA
  test_ref: "HEAD"              # default; override to compare arbitrary refs
  arch: "native"                # CMAKE_CUDA_ARCHITECTURES; "native" detects GPU
  launch_args: "--cuda 13.2 --host gcc14"  # passed to .devcontainer/launch.sh
  nvbench_args: >-
    --timeout 30
    --skip-time 15e-6
    --stopping-criterion entropy
  nvbench_compare_args: ""
```
