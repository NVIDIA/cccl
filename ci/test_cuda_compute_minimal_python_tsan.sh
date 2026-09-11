#!/usr/bin/env bash

# ThreadSanitizer variant of the minimal free-threaded cuda.compute lane.
#
# Installs the TSan-instrumented cuda_cccl wheel (produced by the `python_tsan`
# build, which compiles c.parallel v1 host code with -fsanitize=thread) and runs
# the free-threaded stress tests + the pytest-run-parallel sweep under the TSan
# runtime. A real data race in c.parallel (e.g. build-owned state mutated at
# launch and shared across threads) fails the nightly here instead of silently
# corrupting results under some interleaving.
#
# Only meaningful on a free-threaded (3.14t) interpreter; the GIL would serialize
# the threads and hide the very races we are looking for.

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version>"

require_free_threaded_interpreter() {
  if ! is_free_threaded_python; then
    echo "ERROR: the TSan lane requires a free-threaded interpreter; got '${py_version}'." >&2
    echo "On a GIL interpreter the sweep serializes and TSan has no signal." >&2
    exit 1
  fi
}

# Instrument c.parallel when this script builds the wheel itself (local runs). In
# CI the wheel is the pre-built `python_tsan` artifact, already instrumented; the
# export is harmless there.
export CCCL_C_PARALLEL_SANITIZE_THREAD=1

# Pin the pip cuda-toolkit wheels to the container's CTK minor (this also sets
# cuda_major_version), like every other Python lane. This lane in particular
# cannot run unpinned: the pip libnvrtc finds its libnvrtc-builtins through its
# own RUNPATH, but under the preloaded TSan runtime dlopen is intercepted and
# that RUNPATH is not honored, so the builtins are only found via the system
# toolkit on LD_LIBRARY_PATH -- which only works when the pip NVRTC minor
# matches the container's. Unpinned, a newer 13.x wheel on PyPI breaks every
# compile with "failed to open libnvrtc-builtins.so.<minor>".
pin_cuda_toolkit "${ctk_mode}"

# Setup Python environment
setup_python_env "${py_version}"
require_free_threaded_interpreter

# Fetch or build the TSan-instrumented cuda_cccl wheel. Under project
# `python_tsan`, get_wheel_artifact_name.sh resolves to the distinct `-tsan`
# artifact, so this never grabs an uninstrumented wheel.
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name=$("$ci_dir/util/workflow/get_wheel_artifact_name.sh")
  "$ci_dir/util/artifacts/download.sh" "${wheel_artifact_name}" "${repo_root}/"
  wheelhouse_dir="${repo_root}/wheelhouse"
else
  "$ci_dir/build_cuda_cccl_python.sh" -py-version "${py_version}"
  wheelhouse_dir="${repo_root}/wheelhouse"
fi

# The minimal-cu* extra installs no JIT backend, which this lane requires:
# numba-cuda-mlir DEEPBIND-preloads its bundled LLVM/MLIR libraries, and
# RTLD_DEEPBIND is incompatible with sanitizer runtimes
# (https://github.com/google/sanitizers/issues/611), so only the backend-free
# (no_numba) tests can run under TSan. pytest-run-parallel drives the
# concurrent sweep.
CUDA_CCCL_WHEEL_PATH="$(ls "${wheelhouse_dir}"/cuda_cccl-*.whl)"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[minimal-cu${cuda_major_version}]"
python -m pip install pytest pytest-xdist pytest-run-parallel

# The instrumented .so links libtsan but keeps it external (auditwheel --exclude),
# so the TSan runtime must be present from process start -- LD_PRELOAD the
# runner's libtsan (same soname/major as the gcc-13 build). Without preload the
# .so fails to load ("cannot allocate memory in static TLS block").
tsan_runtime="$(gcc -print-file-name=libtsan.so.2)"
if [[ ! -e "${tsan_runtime}" ]]; then
  echo "ERROR: libtsan.so.2 not found (gcc -print-file-name returned '${tsan_runtime}')." >&2
  exit 1
fi

# setarch -R disables ASLR for the process tree. Required: TSan reserves fixed
# shadow-memory regions and aborts ("unexpected memory mapping") when ASLR drops
# something into them (google/sanitizers#1686). Uses personality(ADDR_NO_RANDOMIZE)
# -- if a runner's seccomp profile blocks it, this call fails and must be allowed
# (e.g. --security-opt seccomp=unconfined on the job's container).
#
# ignore_noninstrumented_modules=1: only c.parallel is instrumented, so ignore
# races inside uninstrumented CPython / CUDA libs (avoids boundary false
# positives). halt_on_error=1: stop at the first race -- it is usually the root
# cause, and later reports are typically downstream noise.
# exitcode=66: exit non-zero when any (unsuppressed) race is found, failing the
# job even though pytest itself passes.
run_under_tsan() {
  setarch -R env \
    LD_PRELOAD="${tsan_runtime}" \
    TSAN_OPTIONS="ignore_noninstrumented_modules=1 halt_on_error=1 history_size=7 exitcode=66" \
    "$@"
}

# Fail fast if the interpreter is not actually GIL-free (wrong build /
# PYTHON_GIL=1): the sweep would run GIL-serialized and pass vacuously.
run_under_tsan python -c "import sys; assert not sys._is_gil_enabled(), 'GIL is enabled; TSan sweep has no signal'"

cd "${repo_root}/python/cuda_cccl/tests/"

# Hand-written free-threaded stress scenarios (they spawn their own worker
# threads) + the serialization node-ids, all sharing one interpreter (-n 0).
run_under_tsan python -m pytest -n 0 -v \
  compute/test_free_threading_stress.py \
  compute/test_multi_cc_serialization.py::test_aot_build_result_load_failure_is_shared_and_retryable \
  compute/test_multi_cc_serialization.py::test_aot_serialization_waits_for_canonical_first_load

# Broad sweep: run the numba-free functional suite with each test executed
# concurrently across threads (barrier-synchronized), exercising many more
# c.parallel algorithms under contention than the hand-written stress tests.
# -n 0 so the threads share one interpreter; --parallel-threads=2 bounds
# GPU-memory pressure and stays reproducible across runners.
# The swept files are the ones marked no_numba module-wide, i.e. everything the
# minimal extras can run.
run_under_tsan python -m pytest -n 0 -v --parallel-threads=2 \
  compute/test_no_numba.py compute/test_raw_op.py
