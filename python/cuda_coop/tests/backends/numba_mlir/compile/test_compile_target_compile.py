# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile generated device code for the kernel's own target.

A kernel can target an architecture other than the current device with
``chip=`` or ``cc=``. Group-method helpers, providers, and Python callbacks
are compiled separately from the kernel and linked into it, so they must use
that same target. Device discovery is fixed to a different compute capability,
and every NVRTC and callback compilation request is recorded while the real
compile and link run.
"""

from types import SimpleNamespace

import pytest
from numba_cuda_mlir import types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_BLOCK_THREADS = 64
_DEVICE_COMPUTE_CAPABILITY = (9, 0)
_LAUNCH_CONFIG_KEY = (
    ("grid", (1, 1, 1)),
    ("block", (_BLOCK_THREADS, 1, 1)),
    ("sharedmem", 0),
    ("cluster", None),
)


def _record_compilations(monkeypatch: pytest.MonkeyPatch):
    """Fix the current device and record coop's compilation requests.

    Return the CUDA module and a namespace whose ``nvrtc`` list holds the
    arguments of each provider and helper compilation and whose ``callbacks``
    list holds the target of each Python callback compilation.
    """

    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda

    from cuda.coop.numba_mlir import _types
    from cuda.coop.numba_mlir._compiler import _nvrtc

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        major, minor = _DEVICE_COMPUTE_CAPABILITY
        return (major, minor) if as_type is tuple else f"sm_{major}{minor}"

    monkeypatch.setattr(
        numba_mlir_tools, "get_gpu_compute_capability", fixed_compute_capability
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=_DEVICE_COMPUTE_CAPABILITY),
    )

    recorded = SimpleNamespace(nvrtc=[], callbacks=[])
    compile_impl = _nvrtc.compile_impl

    def recording_compile_impl(*args, **kwargs):
        recorded.nvrtc.append(kwargs)
        return compile_impl(*args, **kwargs)

    compile_callback = cuda.compile

    def recording_compile_callback(*args, **kwargs):
        recorded.callbacks.append(kwargs.get("cc"))
        return compile_callback(*args, **kwargs)

    monkeypatch.setattr(_nvrtc, "compile_impl", recording_compile_impl)
    monkeypatch.setattr(cuda, "compile", recording_compile_callback)
    # Start without previously compiled callbacks so each one is recorded.
    monkeypatch.setattr(_types, "_DEVICE_LTOIR_CACHE", {})
    return cuda, recorded


def _compile_kernel(cuda, **jit_options):
    """Compile a kernel with a group method, a provider, and a callback.

    ``block.rank()`` becomes a generated helper. The scan with explicit
    ``TempStorage`` is a provider whose storage layout comes from NVRTC.
    The ``scan_op`` device function is compiled as a separate callback.
    """

    import cuda.coop.numba_mlir as coop

    @cuda.jit(device=True)
    def maximum(left, right):
        return left if left > right else right  # noqa: FURB136

    @cuda.jit(**jit_options)
    def kernel(source, output, ranks):
        block = coop.this_block()
        rank = block.rank()
        storage = coop.TempStorage(sharing="shared", auto_sync=True)
        value = coop.inclusive_sum(block, source[rank], temp_storage=storage)
        output[rank] = coop.inclusive_scan(block, value, scan_op=maximum)
        ranks[rank] = rank

    signature = types.void(
        types.int32[::1], types.int32[::1], types.uint32[::1]
    )
    return kernel._compile_launch_config_signature(
        signature, _LAUNCH_CONFIG_KEY
    )


@pytest.mark.parametrize(
    ("jit_options", "arch", "callback_cc"),
    [
        ({"chip": "sm_80"}, "80", (8, 0)),
        ({"cc": (8, 6)}, "86", (8, 6)),
        ({"chip": "sm_90a"}, "90a", (9, 0)),
        ({"chip": "sm_100f"}, "100f", (10, 0)),
    ],
)
def test_generated_code_targets_the_kernel_not_the_current_device(
    monkeypatch: pytest.MonkeyPatch, jit_options, arch, callback_cc
) -> None:
    cuda, recorded = _record_compilations(monkeypatch)

    result = _compile_kernel(cuda, **jit_options)

    assert result.metadata["cubin"]
    assert recorded.nvrtc
    for request in recorded.nvrtc:
        assert request["cc"] == callback_cc[0] * 10 + callback_cc[1]
        assert (
            f"--gpu-architecture=compute_{arch}".encode()
            in request["compiler_options"]
        )
    helper_sources = [
        request["cpp"]
        for request in recorded.nvrtc
        if "cuda_coop_numba_mlir_group_" in request["cpp"]
    ]
    assert helper_sources
    assert all(f"_cc{arch}_ctx_" in source for source in helper_sources)
    assert recorded.callbacks == [callback_cc]
