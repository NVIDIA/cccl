# Tool index — cccl-infra

## Owned (canonical reference lives here)

| Tool | Purpose |
|------|---------|
| `ci/update_version.sh` | Updates `cccl-version.json` and version header files to a new version string. Rejects downgrades. |
| `ci/generate_version.sh` | Regenerates `version.h` and `version.cuh` header files from `cccl-version.json`. |
| `ci/update_rapids_version.sh` | Updates RAPIDS-specific version constraints in `ci/rapids/`. |
| `ci/util/memmon.sh` | Memory usage monitor: polls RSS + swap at configurable intervals and logs peaks. Used in `build_common.sh` for CI build monitoring. |
| `ci/util/retry.sh` | Retries a command N times with configurable backoff. Used in CI for flaky network operations. |
| `ci/util/version_compare.sh` | Compares two semantic version strings (`X.Y.Z`). Used in CI scripts for version guards. |
| `ci/util/extract_switches.sh` | Extracts boolean flag arguments from a command line (e.g. `-lid0`, `-no-lid`). Sourced by per-project build scripts. |
| `ci/util/manifest.sh` | Creates and validates build artifact manifests for CI artifact tracking. |
| `ci/util/create_mock_job_env.sh` | Creates a mock GHA job environment for local testing of CI scripts. |
| `ci/pyenv_helper.sh` | Manages Python virtual environments for CCCL CI (installs packages, activates venvs). |
| `ci/verify_codegen_libcudacxx.sh` | Verifies libcudacxx code generation output for all supported architectures. |
| `ci/install_packaging.sh` | Installs CPM and packaging dependencies for downstream consumption tests. |
| `ci/install_cccl.sh` | Installs CCCL headers and Python wheels to a target system path. |
| `ci/nvrtc_libcudacxx.sh` | Verifies libcudacxx compiles via NVRTC (runtime compilation path). |

## Notes

- `ci/util/memmon.sh` is the only user-visible monitoring utility; it can be enabled locally
  by setting `MEMMON=1` in the environment when invoking a build script.
- `ci/util/retry.sh` and `ci/util/extract_switches.sh` are sourced libraries, not standalone tools.
- Version scripts (`update_version.sh`, `generate_version.sh`) should be invoked via the
  `update-branch-version.yml` workflow rather than directly — the workflow handles branching and PR creation.
