#!/usr/bin/env bash
# Thin wrapper around build_cuda_cccl_python.sh that builds the cuda_cccl
# wheel against the HostJIT-based cccl.c.parallel.v2 library instead of the
# legacy NVRTC v1 library. The shared build script honors CCCL_PYTHON_USE_V2.
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CCCL_PYTHON_USE_V2=1
exec "$ci_dir/build_cuda_cccl_python.sh" "$@"
