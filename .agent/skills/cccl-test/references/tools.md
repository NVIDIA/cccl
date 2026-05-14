# Tool index — cccl-test

## Used (canonical reference lives in cccl-build)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/util/build_and_test_targets.sh` | Targeted build+test driver; `--ctest-targets` and `--lit-tests` flags run the test phase. | `cccl-build` → `references/build_and_test_targets_usage.md` |
| `ci/build_common.sh` | Sourced by full-matrix test scripts for option parsing, compiler validation, `test_preset` helper. | `cccl-build` → `references/build_common.sh_usage.md` |

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/test_cub.sh` | Full-matrix CUB test: host/std/arch sweep; requires GPU. | inherits options from `build_common.sh_usage.md` |
| `ci/test_thrust.sh` | Full-matrix Thrust test. | inherits options from `build_common.sh_usage.md` |
| `ci/test_libcudacxx.sh` | Full-matrix libcudacxx test via lit + ctest. | inherits options from `build_common.sh_usage.md` |
| `ci/test_cudax.sh` | Full-matrix cudax test (C++20 only). | inherits options from `build_common.sh_usage.md` |
| `ci/test_cccl_c_parallel.sh` | C Parallel Library test. | inherits options from `build_common.sh_usage.md` |
| `ci/test_cccl_c_parallel_hostjit.sh` | C Parallel hostjit variant test. | inherits options from `build_common.sh_usage.md` |
| `ci/test_cccl_c_stf.sh` | CCCL C STF test. | inherits options from `build_common.sh_usage.md` |
| `ci/test_packaging.sh` | CPM-based downstream consumption test (no GPU required). | inherits options from `build_common.sh_usage.md` |
| `ci/test_nvbench_helper.sh` | Helper for nvbench benchmark execution during test phase. | inherits options from `build_common.sh_usage.md` |
| `ci/test_python_common.sh` | Shared Python test utilities; sourced by `test_cuda_*.sh` scripts. | sourced library, not invoked directly |
| `ci/test_cuda_compute_python.sh` | cuda.compute Python bindings test. | see `cccl-python` → `references/tools.md` |
| `ci/test_cuda_coop_python.sh` | cuda.cooperative Python bindings test. | see `cccl-python` → `references/tools.md` |
| `ci/test_cuda_cccl_headers_python.sh` | cuda.cccl Python C++ header generation/compilation test. | see `cccl-python` → `references/tools.md` |
| `ci/test_cuda_cccl_examples_python.sh` | cuda.cccl Python example script test. | see `cccl-python` → `references/tools.md` |

## Used

| Tool | Purpose | Reference |
|------|---------|-----------|
| `.devcontainer/launch.sh` | Spin up the devcontainer for GPU test execution. | `cccl-devcontainer` → `references/tools.md` |
