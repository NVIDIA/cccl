#!/usr/bin/env bash
# Run the minimal cuda.compute suite against a wheel built with the v2
# (HostJIT) backend. The shared minimal script owns dependency installation and
# selects the free-threading stress and serialization tests for Python 3.14t.
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CCCL_PYTHON_USE_V2=1
exec "$ci_dir/test_cuda_compute_minimal_python.sh" "$@"
