# Tool index — cccl-ci

## Owned (CI-internal; not user-invoked directly)

These scripts run inside GitHub Actions jobs and are not meant for direct use. They are documented here for diagnostic and maintenance purposes.

| Tool | Purpose |
|------|---------|
| `ci/run_gpu_target.sh` | Entry point for GPU CI jobs: sets environment, launches devcontainer, evaluates the job command, uploads results. |
| `ci/run_cpu_target.sh` | Entry point for CPU-only CI jobs (e.g. static analysis, packaging tests). |
| `ci/run_gpu_bisect.sh` | Entry point for GPU bisect jobs dispatched from `.github/workflows/git-bisect.yml`. |
| `ci/run_cpu_bisect.sh` | Entry point for CPU bisect jobs. |
| `ci/pretty_printing.sh` | Sourced by build/test scripts: colorized `begin_group`/`end_group` banners, `run_command` with retry, `print_var_values`. |
| `ci/upload_cub_test_artifacts.sh` | Packages and uploads CUB test artifacts (binaries + metadata) for multi-runner test jobs. |
| `ci/upload_thrust_test_artifacts.sh` | Packages and uploads Thrust test artifacts. |
| `ci/upload_job_result_artifacts.sh` | Writes a success/fail record for the `workflow-results` aggregation step. Called unconditionally at job end. |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/inspect_changes.py` | Classifies dirty projects from changed paths; drives job pruning. | `cccl_detail-ci` → `references/inspect_changes_usage.md` |
| `ci/util/build_and_test_targets.sh` | Build+test driver called inside CI job containers. | `cccl-build` → `references/build_and_test_targets_usage.md` |
| `ci/util/git_bisect.sh` | Automated bisect invoked by `run_gpu_bisect.sh`. | `cccl-bisect` → `references/git_bisect_usage.md` |
