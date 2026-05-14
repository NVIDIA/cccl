# Tool index — cccl_detail-ci

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/inspect_changes.py` | Classifies dirty CCCL subprojects between two commits or from an explicit file list. Drives job pruning in PR CI. | `references/inspect_changes_usage.md` |
| `ci/ninja_summary.py` | Parses `.ninja_log` to produce a weighted build-time summary (elapsed / weighted by concurrency). Called in `ci/build_common.sh` on CI builds. | no dedicated usage doc; run `ci/ninja_summary.py -h` |
| `ci/test/inspect_changes/regenerate_outputs.sh` | Regenerates expected-output baselines for `inspect_changes.py`'s test suite under `ci/test/inspect_changes/`. | run from repo root; no flags |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/build_common.sh` | Sourced by all `ci/build_*.sh` scripts; defines `build_preset`, `test_preset`, `run_ci_timed_command`, etc. | `cccl-build` → `references/build_common.sh_usage.md` |
