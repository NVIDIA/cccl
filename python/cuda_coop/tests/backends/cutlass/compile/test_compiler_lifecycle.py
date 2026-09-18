# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Real CUTLASS tracing, compilation, linking, and retry lifecycle coverage."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import get_current_env_manager

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core.api._dispatch import _backend_module_name
from cuda.coop.cutlass._compiler import _bundle, _cache
from tests.backends.cutlass.support import device_array, values_for
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile, pytest.mark.gpu]

_THREADS = 32
_ITEMS = 2
_TILE = _THREADS * _ITEMS


def _copy_launcher(api, *, nested=False):
    @cute.jit
    def copy_tile(source: cute.Pointer, destination: cute.Pointer):
        payload = api.ThreadData(_ITEMS)
        api.load(api.this_block(), source, payload)
        api.store(api.this_block(), destination, payload)

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        if cutlass.const_expr(nested):
            copy_tile(source, destination)
        else:
            payload = api.ThreadData(_ITEMS)
            api.load(api.this_block(), source, payload)
            api.store(api.this_block(), destination, payload)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_THREADS)

    return launch


def _exercise_copy(api_name, *, nested=False, repeats=1):
    api = coop if api_name == "common" else cutlass_coop
    launch = _copy_launcher(api, nested=nested)
    source = values_for(np.int32, _TILE, shift=31)
    for _ in range(repeats):
        destination = np.full(_TILE, -101, dtype=np.int32)
        with device_array(source) as src, device_array(destination) as dst:
            compiled = cute.compile(launch, src, dst)
            assert get_current_env_manager() is None
            assert _backend_module_name() is None
            compiled(src, dst)
        np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("api", ("common", "qualified"))
@pytest.mark.parametrize(
    "order",
    ("cutlass-first", "root-first", "root-first-register", "cutlass-first-register"),
)
def test_fresh_process_traces_with_either_import_order(tmp_path, api, order):
    if order.endswith("-register"):
        compiler_import = (
            "import cutlass.cute" if order.startswith("cutlass-first") else ""
        )
        imports = f"""
        import os
        os.environ["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
        {compiler_import}
        from cuda import coop
        assert "cuda.coop.cutlass" not in sys.modules
        assert coop.register("cutlass") is None
        assert "cuda.coop.cutlass" in sys.modules
        """
    elif order == "root-first":
        imports = """
        import cuda.coop
        assert 'cutlass' not in sys.modules
        import cuda.coop.cutlass
        """
    else:
        imports = """
        import cutlass.cute
        import cuda.coop
        assert 'cuda.coop.cutlass' in sys.modules
        """
    script = tmp_path / "fresh_trace.py"
    script.write_text(
        "import sys\n"
        + textwrap.dedent(imports)
        + "from tests.backends.cutlass.compile.test_compiler_lifecycle import _exercise_copy\n"
        + f"_exercise_copy({api!r})\n"
    )
    env = os.environ.copy()
    env.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("api", ("common", "qualified"))
@pytest.mark.parametrize("nested", (False, True), ids=("direct", "nested-jit"))
def test_repeated_compile_and_nested_jit_leave_no_active_backend(api, nested):
    _exercise_copy(api, nested=nested, repeats=2)


@pytest.mark.parametrize("failure", ("provider-compile", "final-link"))
def test_failed_compilation_or_linking_can_retry(monkeypatch, tmp_path, failure):
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
                cute.compile(launch, src, dst)
        assert attempted
        if failure == "provider-compile":
            assert "injected provider compilation failure" in str(caught.value)
        else:
            message = str(caught.value).lower()
            assert any(
                token in message for token in ("link", "lto", "nvvm", "compile")
            ), message
        assert get_current_env_manager() is None
        assert _backend_module_name() is None
        compiled = cute.compile(launch, src, dst)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)


def test_failed_trace_after_registering_a_provider_can_retry():
    @cute.kernel
    def kernel(
        source: cute.Pointer, destination: cute.Pointer, reject: cutlass.Constexpr
    ):
        payload = coop.ThreadData(_ITEMS)
        coop.load(coop.this_block(), source, payload)
        if cutlass.const_expr(reject):
            raise ValueError("injected failure after provider registration")
        coop.store(coop.this_block(), destination, payload)

    @cute.jit
    def launch(
        source: cute.Pointer, destination: cute.Pointer, reject: cutlass.Constexpr
    ):
        kernel(source, destination, reject).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _TILE, shift=61)
    destination = np.full(_TILE, -101, dtype=np.int32)
    with device_array(source) as src, device_array(destination) as dst:
        with pytest.raises(ValueError, match="after provider registration"):
            cute.compile(launch, src, dst, True)
        assert get_current_env_manager() is None
        assert _backend_module_name() is None
        compiled = cute.compile(launch, src, dst, False)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)


def test_unrelated_trace_does_not_compile_or_relink_a_provider(monkeypatch):
    _exercise_copy("common")

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
