# Tool index — cccl-python

## Owned (canonical reference lives here)

| Tool | Purpose | Detail |
|------|---------|--------|
| `ci/util/python/common_arg_parser.sh` | Shared argument parsing utilities sourced by Python CI scripts. Provides common flags (`--cuda-version`, `--python-version`, etc.) used across `test_cuda_*.sh` scripts. | sourced library, not invoked directly |

## Used (canonical reference lives in another skill)

| Tool | Purpose | Reference |
|------|---------|-----------|
| `ci/build_cuda_cccl_wheel.sh` | Builds the cuda-cccl Python wheel package. | `cccl-build` → `references/tools.md` |
| `ci/build_cuda_cccl_python.sh` | Builds cuda.cccl in-tree (dev install mode). | `cccl-build` → `references/tools.md` |
| `ci/test_cuda_compute_python.sh` | Tests cuda.compute Python bindings. | `cccl-test` → `references/tools.md` |
| `ci/test_cuda_coop_python.sh` | Tests cuda.cooperative Python bindings. | `cccl-test` → `references/tools.md` |
| `ci/test_cuda_cccl_headers_python.sh` | Tests cuda.cccl Python C++ header generation and compilation. | `cccl-test` → `references/tools.md` |
| `ci/test_cuda_cccl_examples_python.sh` | Tests cuda.cccl Python example scripts. | `cccl-test` → `references/tools.md` |
