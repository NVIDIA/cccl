#!/usr/bin/env bash
# Thin wrapper around build_cuda_cccl_python.sh that builds the cuda_cccl wheel
# with ThreadSanitizer instrumentation on the c.parallel (v1) host code, for the
# free-threaded (3.14t) TSan nightly lane. The shared build script honors
# CCCL_C_PARALLEL_SANITIZE_THREAD (passes -DCCCL_C_PARALLEL_SANITIZE_THREAD=ON to
# the wheel build and --exclude libtsan from the auditwheel repair).
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CCCL_C_PARALLEL_SANITIZE_THREAD=1
exec "$ci_dir/build_cuda_cccl_python.sh" "$@"
