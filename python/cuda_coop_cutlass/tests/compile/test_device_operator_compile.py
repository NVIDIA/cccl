# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile scalar callback signatures and recover from failed final links."""

from dataclasses import replace

import cutlass
import pytest
from cutlass import cute
from cutlass.base_dsl.common import get_current_env_manager
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._compiler import _bundle, _cache

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
    with pytest.raises(Exception, match="targets.*sm_90|architecture"):
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
