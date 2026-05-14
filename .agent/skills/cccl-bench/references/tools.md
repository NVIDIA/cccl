# Tool index — cccl-bench

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/bench/bench.sh` | Thin wrapper: compares two git refs by forwarding to `compare_git_refs.sh`. Usage: `bench.sh <base-ref> <test-ref> [compare_paths args...]`. | see `references/local-run.md` |
| `ci/bench/compare_git_refs.sh` | Builds both refs and compares their benchmark output. Core benchmark comparison driver. | see `references/local-run.md` |
| `ci/bench/compare_paths.sh` | Compares benchmark results from two pre-built paths (skips build step). | see `references/local-run.md` |
| `ci/bench/parse_bench_matrix.sh` | Parses `ci/bench.yaml` to extract benchmark job definitions for CI dispatch. | CI-internal; not user-invoked |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/util/build_and_test_targets.sh` | Used internally by bench scripts to build benchmark targets before running. | `cccl-build` → `references/build_and_test_targets_usage.md` |
| `.devcontainer/launch.sh` | Wraps bench runs in the devcontainer. | `cccl-devcontainer` → `references/launch_usage.md` |
