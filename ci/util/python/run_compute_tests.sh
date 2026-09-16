#!/usr/bin/env bash
# Test payload: the cuda.compute pytest suite.
# Invoked by ci/test_cuda_compute_python.sh, which has already
# put the cuda_cccl wheel in wheelhouse/.
#
# Runs in the minimal container: nothing here may assume more than Python and
# the wheel's declared deps (docs/infrastructure/ci/references/ci_scripts.rst).

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

python_payload_init "$@"

# Install cuda_cccl. The extra flavor is "cu" (pip-installed toolkit) or "sysctk"
# (system-provided toolkit) depending on the -ctk-mode arg.
CUDA_CCCL_WHEEL_PATH="$(cuda_cccl_wheel_path)"
ctk_flavor="$(ctk_extra_flavor "${ctk_mode}")"
python -m pip install "${CUDA_CCCL_WHEEL_PATH}[test-${ctk_flavor}${cuda_major_version}]"

# On the v2 (HostJIT) backend, abort on first failure — the suite is still
# stabilizing and a single early failure is enough signal to investigate
# without scrolling through hundreds of subsequent passes.
pytest_extra=()
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  pytest_extra+=(-x)
fi

cd "${repo_root}/python/cuda_cccl/tests/"
if [[ "${CCCL_PYTHON_USE_V2:-}" =~ ^(1|true|TRUE|on|ON)$ ]] && ! is_free_threaded_python; then
  # The test isolates itself in a fresh subprocess (LLVM initialization is
  # process-wide and only cold once), but it carries the free_threading marker,
  # so on a GIL interpreter it must be selected by node-id or nothing runs it.
  # On a free-threaded one the marker-selected block below covers it already.
  python -m pytest "${pytest_extra[@]}" -n 0 -v \
    compute/test_free_threading_stress.py::test_v2_concurrent_cold_llvm_initialization
fi
python -m pytest "${pytest_extra[@]}" -n 6 -v compute/ -m "not large and not free_threading"
python -m pytest "${pytest_extra[@]}" -n 0 -v compute/ -m "large and not free_threading"

# The free-threading suites carry the free_threading marker, so the sweeps above
# exclude them -- on a GIL interpreter they would only skip themselves. Run them
# here when the interpreter is genuinely free-threaded. On the full extras this
# exercises test_free_threading_stress.py with Python-callable operators,
# gpu_struct types and unannotated TransformIterator available, which the
# minimal payload cannot.
if is_free_threaded_python; then
  # Fail loudly if the GIL is on anyway (wrong build / PYTHON_GIL=1) instead of
  # letting each stress test fail separately with a less obvious message.
  python -c "import sys; assert not sys._is_gil_enabled(), 'GIL is enabled; free-threading tests have no signal'"

  # -n 0: these spawn and barrier-synchronize their own worker threads, so
  # pytest itself must stay in a single process. Selected by marker rather than
  # by filename so a future free-threading suite is picked up automatically.
  python -m pytest "${pytest_extra[@]}" -n 0 -v -m free_threading compute/

  # Broad thread-safety sweep: re-run the functional suite with each test
  # executed concurrently across threads (barrier-synchronized start), which
  # stresses the process-wide build cache, single-flight coordination and the
  # Cython bindings from many threads at once. Complements the hand-written
  # suites above, which target specific shared-object scenarios.
  #
  # Same selector as the -n 6 run above. "not large" because those tests exist
  # to make big device allocations and are already run at -n 0 for that reason
  # (see the split above) -- sweeping them re-creates the memory pressure that
  # split exists to avoid. "not free_threading" because they just ran, with
  # their own workers; they also carry pytest.mark.thread_unsafe, so the plugin
  # would demote them to one thread here anyway.
  #
  # --parallel-threads=2 matches CuPy's free-threading CI (the closest GPU
  # precedent); a small fixed count bounds GPU-memory pressure and stays
  # reproducible across runners, unlike =auto (the runner's core count).
  #
  # pytest-run-parallel is only used by this sweep, so install it here rather
  # than carrying it in the test extras.
  python -m pip install pytest-run-parallel
  python -m pytest "${pytest_extra[@]}" -n 0 -v --parallel-threads=2 \
    compute/ -m "not large and not free_threading"
fi

# The bfloat16 tests require ml_dtypes (the NumPy bfloat16 extension dtype),
# which is deliberately not part of the test extras so that the sweeps above
# run in an environment matching a user's default install (where the bfloat16
# tests skip themselves). Install it last and run those tests explicitly.
python -m pip install ml_dtypes
python -m pytest "${pytest_extra[@]}" -n 6 -v compute/test_bfloat16.py
