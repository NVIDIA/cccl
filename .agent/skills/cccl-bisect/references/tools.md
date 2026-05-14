# Tool index — cccl-bisect

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/util/git_bisect.sh` | Automated git bisect: checks out commits, runs build+test via the same flags as `build_and_test_targets.sh`, and reports the first bad commit. | `references/git_bisect_usage.md` |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/util/build_and_test_targets.sh` | Build/test driver invoked internally by `git_bisect.sh` for each bisect step. | `cccl-build` → `references/build_and_test_targets_usage.md` |
| `.devcontainer/launch.sh` | Wraps `git_bisect.sh` in the devcontainer for local bisect runs. | `cccl-devcontainer` → `references/tools.md` |
