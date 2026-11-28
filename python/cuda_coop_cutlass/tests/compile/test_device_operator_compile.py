# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile scalar callback signatures and recover from failed final links."""

import subprocess
import sys
from dataclasses import replace

import cutlass
import numpy as np
import pytest
from cuda.pathfinder import find_nvidia_binary_utility
from cutlass import cute
from cutlass.base_dsl.common import DSLUserCodeError, get_current_env_manager
from cutlass.base_dsl.compiler import (
    DumpDir,
    GPUArch,
    KeepCUBIN,
    LinkLibraries,
    PtxasOptions,
)
from cutlass.compiler import LlvmIrArtifact
from cutlass.cute.ffi import extern
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import CuTeDSL

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._compiler import _bundle, _cache, _finalize, _nvrtc
from cuda.coop.cutlass._compiler._windows_pipeline import configure_device_link
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.compile]


@cute.jit
def _add_one(a, b) -> cutlass.Int32:
    return a + b + 1


def _launcher(callback, dtype=cutlass.Int32):
    @cute.kernel
    def kernel(destination: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        result = coop.reduce(
            coop.this_block(), dtype(thread), binary_op=callback
        )
        if thread == 0:
            cute.make_tensor(destination, cute.make_layout(1))[0] = result

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    return launch


def _compile(launch, dtype=cutlass.Int32):
    pointer = make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile[(GPUArch("sm_80"),)](launch, pointer)


@pytest.fixture
def windows_device_link(monkeypatch):
    if sys.platform == "win32":
        return
    if _nvrtc.resolve_compile_context(()).nvjitlink_version < (13, 4):
        pytest.skip(
            "The Windows device linker requires nvJitLink 13.4 or later"
        )
    # The forced linker replaces the pass manager on CuTe's singleton.
    compiler = CuTeDSL().compiler_provider
    monkeypatch.setattr(compiler, "passmanager", compiler.passmanager)
    monkeypatch.setattr(_finalize, "_attach_link_inputs", configure_device_link)


@pytest.mark.usefixtures("windows_device_link")
@pytest.mark.parametrize("arch", ("sm_80", "sm_90a"))
def test_windows_device_link_targets(tmp_path, arch):
    callback = coop.compile_operator(_add_one, cutlass.Int32, arch=arch)
    pointer = make_ptr(
        cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16
    )
    options = (GPUArch(arch), KeepCUBIN, DumpDir(str(tmp_path)))
    cute.compile[options](_launcher(callback), pointer)
    (cubin,) = tmp_path.glob("*.cubin")
    assert arch in cubin.name
    assert cubin.read_bytes().startswith(b"\x7fELF")
    cuobjdump = find_nvidia_binary_utility("cuobjdump")
    if cuobjdump is not None:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        assert f"code for {arch}\n" in sass


@pytest.mark.gpu
@pytest.mark.usefixtures("windows_device_link")
def test_windows_device_link_mixed_libraries(tmp_path):
    """Run one kernel that calls user bitcode, user LTO and a CCCL scan."""
    callback = coop.compile_operator(_add_one, cutlass.Int32)
    library_dir = tmp_path / "user λ"
    library_dir.mkdir()
    lto = library_dir / "combine.ltoir"
    lto.write_bytes(callback._ltoir)
    bitcode = tmp_path / "triple.bc"
    module = LlvmIrArtifact.from_textual_form(
        b"define i32 @triple(i32 %x) { %y = mul i32 %x, 3 ret i32 %y }"
    )
    bitcode.write_bytes(module.get_bitcode())

    @extern(name=callback.symbol)
    def combine(a: cutlass.Int32, b: cutlass.Int32) -> cutlass.Int32: ...

    @extern
    def triple(x: cutlass.Int32) -> cutlass.Int32: ...

    @cute.kernel
    def kernel(destination: cute.Pointer):
        block = coop.this_block()
        thread = block.rank()
        output = cute.make_tensor(destination, cute.make_layout(32))
        value = combine(triple(cutlass.Int32(thread)), cutlass.Int32(0))
        output[thread] = coop.inclusive_scan(block, cutlass.Int32(value))

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    options = (
        LinkLibraries(f"{bitcode},{lto}"),
        PtxasOptions("--maxrregcount=32 --warn-on-spills"),
        KeepCUBIN,
        DumpDir(str(tmp_path)),
    )
    result = np.full(32, -1, dtype=np.int32)
    with device_array(result) as dst:
        compiled = cute.compile[options](launch, dst)
        compiled(dst)
    np.testing.assert_array_equal(result, np.cumsum(3 * np.arange(32) + 1))
    (cubin,) = tmp_path.glob("*.cubin")
    assert cubin.read_bytes().startswith(b"\x7fELF")


@pytest.mark.parametrize("failure", ("provider-compile", "final-link"))
def test_failed_compile_or_link_can_retry(monkeypatch, tmp_path, failure):
    callback = coop.compile_operator(_add_one, cutlass.Int32, arch="sm_80")
    launch = _launcher(callback)
    original = _bundle.compile_bundle_source_with_layouts
    attempted = []
    malformed = tmp_path / "invalid-provider.ltoir"
    malformed.write_bytes(b"This is deliberately not NVIDIA LTO-IR.\n")

    def fail_bundle(*args, **kwargs):
        attempted.append(True)
        if failure == "provider-compile":
            raise RuntimeError("injected callback provider compilation failure")
        compilation = original(*args, **kwargs)
        _cache.add_managed_bundle_path(str(malformed))
        return replace(compilation, path=str(malformed))

    with monkeypatch.context() as patch:
        patch.setattr(
            _bundle, "compile_bundle_source_with_layouts", fail_bundle
        )
        with pytest.raises(Exception) as caught:
            _compile(launch)
    assert attempted
    if failure == "provider-compile":
        assert "injected callback provider compilation failure" in str(
            caught.value
        )
    else:
        message = str(caught.value).lower()
        assert any(
            token in message for token in ("link", "lto", "nvvm", "compile")
        ), message
    assert get_current_env_manager() is None
    assert _compile(launch) is not None
    assert get_current_env_manager() is None


def test_callback_dtype_must_match_collective():
    callback = coop.compile_operator(_add_one, cutlass.Int32, arch="sm_80")
    with pytest.raises(Exception, match="dtype|type.*match"):
        _compile(_launcher(callback, cutlass.Float32), cutlass.Float32)


def test_unprepared_callback_is_rejected():
    with pytest.raises(Exception, match="compile_operator|custom callbacks"):
        _compile(_launcher(_add_one))


@pytest.mark.parametrize(
    "signature", ("one-argument", "return-dtype", "return-body")
)
def test_callback_signature_must_be_binary_and_closed(signature):
    @cute.jit
    def unary(a) -> cutlass.Int32:
        return a

    @cute.jit
    def wrong_return(a, b) -> cutlass.Float32:
        return cutlass.Float32(a + b)

    @cute.jit
    def wrong_body(a, b) -> cutlass.Int32:
        return cutlass.Float32(a + b)

    callback = {
        "one-argument": unary,
        "return-dtype": wrong_return,
        "return-body": wrong_body,
    }[signature]
    message = (
        "two|binary|argument"
        if signature == "one-argument"
        else ("return|result|dtype")
    )
    with pytest.raises(Exception, match=message):
        coop.compile_operator(callback, cutlass.Int32, arch="sm_80")
    assert get_current_env_manager() is None


def test_callback_architecture_must_match_kernel():
    callback = coop.compile_operator(_add_one, cutlass.Int32, arch="sm_90")
    with pytest.raises(DSLUserCodeError, match="targets.*sm_90|architecture"):
        _compile(_launcher(callback))


@pytest.mark.parametrize(
    "profile, message",
    (("commutative-only", "(?i)commutativ"), ("batched", "reduce_batched")),
)
def test_unsupported_callback_profiles(profile, message):
    callback = coop.compile_operator(_add_one, cutlass.Int32, arch="sm_80")

    @cute.kernel
    def kernel(destination: cute.Pointer):
        if cutlass.const_expr(profile == "commutative-only"):
            coop.reduce(
                coop.this_block(),
                cutlass.Int32(1),
                binary_op=callback,
                algorithm="raking_commutative_only",
            )
        else:
            values = coop.ThreadData(
                1, dtype=cutlass.Int32, values=[cutlass.Int32(1)]
            )
            coop.reduce_batched(coop.this_warp(), values, binary_op=callback)

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    with pytest.raises(Exception, match=message):
        _compile(launch)


def test_callback_preparation_requires_host_context():
    @cute.kernel
    def kernel(destination: cute.Pointer):
        coop.compile_operator(_add_one, cutlass.Int32, arch="sm_80")

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    with pytest.raises(
        Exception, match="outside.*trac|during.*trac|active.*trac|before.*trac"
    ):
        _compile(launch)


@pytest.mark.parametrize("arch", ("sm_nope", "sm_90aa", "compute_90x"))
def test_invalid_callback_arch(arch):
    with pytest.raises(DSLUserCodeError, match="Invalid configured CUDA arch"):
        coop.compile_operator(_add_one, cutlass.Int32, arch=arch)
