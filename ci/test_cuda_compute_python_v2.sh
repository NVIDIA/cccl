#!/usr/bin/env bash
# Run the cuda.compute pytest suite against a wheel built with the v2
# (HostJIT) backend. Mirrors test_cuda_compute_python.sh; the only difference
# is exporting CCCL_PYTHON_USE_V2 so the wheel build (and downstream pytest)
# uses cccl.c.parallel.v2.
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CCCL_PYTHON_USE_V2=1
exec "$ci_dir/test_cuda_compute_python.sh" "$@"
