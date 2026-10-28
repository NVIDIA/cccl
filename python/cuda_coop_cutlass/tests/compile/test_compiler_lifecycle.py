# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercise compiler environment and provider cleanup through real GPU copies.

Fresh processes test import order without an already populated module cache.
Repeated and nested traces must release their compiler environment. Injected
failures at tracing, provider compilation, and final linking must leave a
subsequent compile usable; unrelated kernels must not inherit provider work.
"""

import subprocess
import sys

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.base_dsl.common import get_current_env_manager

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._compiler import _bundle, _cache
from tests.runtime_support import device_array, values_for
from tests.support.paths import TESTS_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile, pytest.mark.gpu]

_THREADS = 32
_ITEMS = 2
_TILE = _THREADS * _ITEMS


def _copy_launcher(api, *, nested=False):
    """Build direct or nested calls with the same cooperative copy behavior.

    The nested form enters another cute.jit function within one kernel trace.
    Both forms share the same input/output contract, so tests can isolate
    provider ownership and cleanup from differences in kernel behavior.
    """

    @cute.jit
    def copy_tile(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        payload = api.load(
            api.this_block(), source, items_per_thread=items_per_thread
        )
        api.store(api.this_block(), destination, payload)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        if cutlass.const_expr(nested):
            copy_tile(source, destination, items_per_thread)
        else:
            payload = api.load(
                api.this_block(), source, items_per_thread=items_per_thread
            )
            api.store(api.this_block(), destination, payload)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, items_per_thread).launch(
            grid=1, block=_THREADS
        )

    return launch


def _exercise_copy(*, nested=False, repeats=1):
    """Compile and run a copy while checking that tracing has ended.

    The context checks occur after compile returns and before GPU execution.
    Each repeat checks copied values from fresh device buffers on the host.
    A successful compile still fails if it leaves a compiler environment
    active.
    """

    launch = _copy_launcher(coop, nested=nested)
    source = values_for(np.int32, _TILE, shift=31)
    for _ in range(repeats):
        destination = np.full(_TILE, -101, dtype=np.int32)
        with device_array(source) as src, device_array(destination) as dst:
            compiled = cute.compile(launch, src, dst, _ITEMS)
            assert get_current_env_manager() is None

            compiled(src, dst)
        np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("order", ("cutlass-first", "coop-first"))
def test_fresh_process_traces_with_either_import_order(tmp_path, order):
    """Import the explicit adapter before or after CuTe in a fresh process."""

    if order == "cutlass-first":
        imports = "import cutlass.cute\nimport cuda.coop.cutlass\n"
    else:
        imports = "import cuda.coop.cutlass\nimport cutlass.cute\n"
    script = tmp_path / "fresh_trace.py"
    script.write_text(
        "import sys, types\n"
        + f"sys.path[:] = {sys.path!r}\n"
        + "tests = types.ModuleType('tests')\n"
        + f"tests.__path__ = {[str(TESTS_ROOT)]!r}\n"
        + "sys.modules['tests'] = tests\n"
        + imports
        + "from tests.compile.test_compiler_lifecycle import _exercise_copy\n"
        + "_exercise_copy()\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("nested", (False, True), ids=("direct", "nested-jit"))
def test_repeated_compile_and_nested_jit_leave_no_active_environment(nested):
    _exercise_copy(nested=nested, repeats=2)


@pytest.mark.parametrize("failure", ("provider-compile", "final-link"))
def test_failed_compilation_or_linking_can_retry(
    monkeypatch, tmp_path, failure
):
    """Retry the launcher after provider compilation or final linking fails.

    One injection fails provider compilation. The other compiles a real
    provider, then supplies invalid bytes to final linking. After restoring
    the hook, require a successful compile and copy with no compiler
    environment left active by the failed attempt.
    """

    launch = _copy_launcher(coop)
    source = values_for(np.int32, _TILE, shift=47)
    destination = np.full(_TILE, -101, dtype=np.int32)
    original = _bundle.compile_bundle_source
    attempted = []
    malformed = tmp_path / "invalid-provider.ltoir"
    malformed.write_bytes(b"This is deliberately not NVIDIA LTO-IR.\n")

    def fail_bundle(*args, **kwargs):
        attempted.append(True)
        if failure == "provider-compile":
            raise RuntimeError("injected provider compilation failure")
        # Generate the real provider first, then fault only the final linker
        # input. The retry must re-trace and link the valid provider artifact.
        original(*args, **kwargs)
        _cache.add_managed_bundle_path(str(malformed))
        return str(malformed)

    with device_array(source) as src, device_array(destination) as dst:
        with monkeypatch.context() as patch:
            patch.setattr(_bundle, "compile_bundle_source", fail_bundle)
            with pytest.raises(Exception) as caught:
                cute.compile(launch, src, dst, _ITEMS)
        assert attempted
        if failure == "provider-compile":
            assert "injected provider compilation failure" in str(caught.value)
        else:
            message = str(caught.value).lower()
            assert any(
                token in message for token in ("link", "lto", "nvvm", "compile")
            ), message
        assert get_current_env_manager() is None

        compiled = cute.compile(launch, src, dst, _ITEMS)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_failed_trace_after_registering_a_provider_can_retry(items_per_thread):
    """Discard provider work from a trace that raises after its first load.

    The rejection occurs after Load registers its provider and before Store.
    The failed trace must release the compiler environment. A retry with
    rejection disabled must complete the copy for both payload extents.
    """

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        reject: cutlass.Constexpr,
        items_per_thread: cutlass.Constexpr,
    ):
        payload = coop.load(
            coop.this_block(), source, items_per_thread=items_per_thread
        )
        if cutlass.const_expr(reject):
            raise ValueError("injected failure after provider registration")
        coop.store(coop.this_block(), destination, payload)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        reject: cutlass.Constexpr,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, destination, reject, items_per_thread).launch(
            grid=1, block=_THREADS
        )

    source = values_for(np.int32, _THREADS * items_per_thread, shift=61)
    destination = np.full(_THREADS * items_per_thread, -101, dtype=np.int32)
    with device_array(source) as src, device_array(destination) as dst:
        with pytest.raises(ValueError, match="after provider registration"):
            cute.compile(launch, src, dst, True, items_per_thread)
        assert get_current_env_manager() is None

        compiled = cute.compile(launch, src, dst, False, items_per_thread)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)


def test_unrelated_trace_does_not_compile_or_relink_a_provider(monkeypatch):
    """Follow a cooperative kernel with an ordinary CuTe kernel.

    The first copy populates provider state. Then install a failing hook to
    expose leaked provider work when the unrelated kernel is traced. Its
    separate arithmetic output confirms that the second kernel still runs.
    """

    _exercise_copy()

    @cute.kernel
    def kernel(destination: cute.Pointer):
        thread, _, _ = cute.arch.thread_idx()
        tensor = cute.make_tensor(destination, cute.make_layout(_THREADS))
        tensor[thread] = thread + 5

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=_THREADS)

    def unexpected_bundle(*args, **kwargs):
        raise AssertionError("unrelated trace requested a cooperative provider")

    monkeypatch.setattr(_bundle, "compile_bundle_source", unexpected_bundle)
    destination = np.full(_THREADS, -101, dtype=np.int32)
    with device_array(destination) as dst:
        compiled = cute.compile(launch, dst)
        compiled(dst)
    np.testing.assert_array_equal(destination, np.arange(_THREADS) + 5)
