# Tool index — cccl-build

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/util/build_and_test_targets.sh` | Targeted configure/build/test driver for a single preset. Wraps cmake, ninja, ctest, lit. | `references/build_and_test_targets_usage.md` |
| `ci/build_common.sh` | Sourced library: option parsing, compiler validation, build dir layout, helper functions for all `ci/build_*.sh` scripts. | `references/build_common.sh_usage.md` |
| `ci/build_cub.sh` | Full-matrix CUB build: host/std/arch sweep; Launch ID (LID) partitioning for CI artifacts. | see `build_common.sh_usage.md` for common options |
| `ci/build_thrust.sh` | Full-matrix Thrust build: host/std/arch sweep. | see `build_common.sh_usage.md` |
| `ci/build_libcudacxx.sh` | Full-matrix libcudacxx build with lit/ctest. | see `build_common.sh_usage.md` |
| `ci/build_cudax.sh` | Full-matrix cudax build (C++20 only). | see `build_common.sh_usage.md` |
| `ci/build_cccl_c_parallel.sh` | Full-matrix C Parallel Library build. | see `build_common.sh_usage.md` |
| `ci/build_cccl_c_parallel_hostjit.sh` | C Parallel hostjit variant build. | see `build_common.sh_usage.md` |
| `ci/build_cccl_c_stf.sh` | CCCL C library STF test build. | see `build_common.sh_usage.md` |
| `ci/build_stdpar.sh` | C++ standard parallel algorithms support build. | see `build_common.sh_usage.md` |
| `ci/build_tidy.sh` | clang-tidy static analysis run across CCCL. | see `build_common.sh_usage.md` |
| `ci/build_cuda_cccl_wheel.sh` | cuda-cccl Python wheel package build. | see `cccl-python` → `references/tools.md` |
| `ci/build_cuda_cccl_python.sh` | cuda.cccl Python package in-tree dev build. | see `cccl-python` → `references/tools.md` |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `.devcontainer/launch.sh` | Spin up or exec into a devcontainer for the build. | `cccl-devcontainer` → `references/tools.md` |

## Notes

`ci/build_cub.sh` accepts `-lid0`, `-lid1`, `-lid2`, `-no-lid` to select the `cub-lid0`, `cub-lid1`,
`cub-lid2`, or `cub-nolid` CMake preset. These correspond to Launch ID (LID) partitions used in CI
to split CUB's large test suite across multiple runners.
