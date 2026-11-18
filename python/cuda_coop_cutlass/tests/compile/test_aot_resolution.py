# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Capture an existing JIT result and distinguish strict and fallback misses."""

import sys
from pathlib import Path

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import GPUArch

import cuda.coop.cutlass as coop
from cuda.coop.cutlass import _aot_pack, aot
from cuda.coop.cutlass._aot_cli import main
from cuda.coop.cutlass._compiler import _bundle, _cache, _nvrtc

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


def test_capture_cached_provider_and_handle_changed_specialization(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv(_cache.CACHE_DIR_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(_cache, "_SOURCE_CACHE", {})
    pack = tmp_path / "pack"

    def compile_provider(value):
        source = f'extern "C" __device__ int value() {{ return {value}; }}'
        return Path(
            _bundle.compile_bundle_source(
                source, arch="compute_90", required_headers=()
            )
        )

    original = compile_provider(1).read_bytes()
    _cache._SOURCE_CACHE.clear()
    with monkeypatch.context() as cached:
        cached.setattr(
            _nvrtc,
            "compile_ltoir",
            lambda *args: pytest.fail("JIT cache missed"),
        )
        with aot.capture(pack):
            assert compile_provider(1).read_bytes() == original

    with (
        aot.use(pack, mode="auto"),
        caplog.at_level("DEBUG", logger="cuda.coop.cutlass._aot_pack"),
    ):
        changed = compile_provider(2).read_bytes()
    assert changed != original
    assert "entry is absent" in caplog.text

    def forbid_jit(*args):
        pytest.fail("Required pack consulted the provider JIT")

    monkeypatch.setattr(_nvrtc, "resolve_compile_context", forbid_jit)
    with aot.use(pack, mode="required"):
        assert compile_provider(1).read_bytes() == original
        # Even though the JIT cache now contains value(2), the pack does not.
        dump_dir = tmp_path / "sources"
        monkeypatch.setenv("CUDA_COOP_CUTLASS_SOURCE_DUMP_DIR", str(dump_dir))
        with pytest.raises(aot.PackMissError):
            compile_provider(2)
        (dump,) = dump_dir.glob("*.cu")
        assert "return 2;" in dump.read_text()


@cute.kernel
def _sum_kernel():
    coop.sum(coop.this_block(), cutlass.Int32(cute.arch.thread_idx()[0]))


@cute.jit
def _sum_launcher():
    _sum_kernel().launch(grid=1, block=32)


def _compile_sum():
    """Trace through the real finalizer with an explicit GPU target."""
    cute.compile[(GPUArch("sm_80"),)](_sum_launcher)


def test_pack_errors_through_cute_and_cli(tmp_path, monkeypatch, capsys):
    pack = tmp_path / "unrelated-pack"
    with aot.capture(pack):
        _aot_pack.record("0" * 64, "compute_80", b"unrelated", {}, (13, 3))

    with aot.use(pack, mode="required"), pytest.raises(DSLRuntimeError) as err:
        _compile_sum()
    assert isinstance(err.value.__cause__, aot.PackMissError)

    workload = tmp_path / "sum.py"
    workload.write_text(
        f"import runpy\nrunpy.run_path({__file__!r})['_compile_sum']()\n"
    )
    assert (
        main(
            [
                "run",
                "--pack",
                str(pack),
                "--mode",
                "required",
                "--",
                sys.executable,
                str(workload),
            ]
        )
        == 1
    )
    assert f"cuda-coop-aot: {err.value.__cause__}" in capsys.readouterr().err

    monkeypatch.setenv(_aot_pack.PACK_PATH_ENV, str(tmp_path / "missing-pack"))
    monkeypatch.setenv(_aot_pack.PACK_MODE_ENV, "required")
    with pytest.raises(DSLRuntimeError) as err:
        _compile_sum()
    assert isinstance(err.value.__cause__, aot.PackIntegrityError)
