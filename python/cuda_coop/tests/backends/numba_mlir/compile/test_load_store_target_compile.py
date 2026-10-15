# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile Load/Store providers for the enclosing kernel's target and options.

Device discovery reports SM90 while kernels explicitly target other chips.
Record actual NVRTC requests and compile/link the complete kernels to verify
architecture suffixes, lineinfo options, and debug-link compatibility.
The SM80 case also relinks the actual inputs explicitly for SM80 because the
compiler can promote its normal final link to the current device's target.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

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
    """Fix device discovery and record provider compilation and link inputs."""

    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.linker import Linker

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

    recorded = SimpleNamespace(nvrtc=[], links=[])
    compile_impl = _nvrtc.compile_impl

    def recording_compile_impl(*args, **kwargs):
        recorded.nvrtc.append(kwargs)
        return compile_impl(*args, **kwargs)

    monkeypatch.setattr(_nvrtc, "compile_impl", recording_compile_impl)
    complete = Linker.complete

    def recording_complete(self):
        result = complete(self)
        recorded.links.append(
            SimpleNamespace(
                arch=self.arch,
                inputs=tuple(self._object_codes),
                cubin=result.code,
            )
        )
        return result

    monkeypatch.setattr(Linker, "complete", recording_complete)
    return cuda, recorded


def _compile_kernel(cuda, items_per_thread, **jit_options):
    """Compile Load/Store providers with explicit shared temporary storage."""

    from cuda import coop

    @cuda.jit(**jit_options)
    def kernel(source, output, items_per_thread):
        block = coop.this_block()
        payload = coop.ThreadData(items_per_thread)
        storage = coop.TempStorage(sharing="shared", auto_sync=True)
        coop.load(
            block, source, payload, algorithm="transpose", temp_storage=storage
        )
        coop.store(
            block, output, payload, algorithm="transpose", temp_storage=storage
        )

    signature = types.void(
        types.int32[::1],
        types.int32[::1],
        types.IntegerLiteral(items_per_thread),
    )
    return kernel._compile_launch_config_signature(
        signature, _LAUNCH_CONFIG_KEY
    )


@pytest.mark.parametrize(
    ("jit_options", "arch", "cc"),
    [
        ({"chip": "sm_80"}, "80", 80),
        ({"cc": (8, 6)}, "86", 86),
        ({"chip": "sm_90a"}, "90a", 90),
        ({"chip": "sm_100f"}, "100f", 100),
    ],
)
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_load_store_targets_the_kernel_not_the_current_device(
    monkeypatch: pytest.MonkeyPatch, items_per_thread, jit_options, arch, cc
) -> None:
    cuda, recorded = _record_compilations(monkeypatch)

    result = _compile_kernel(cuda, items_per_thread, **jit_options)

    assert result.metadata["cubin"]
    if arch == "80":
        from cuda.core import Linker, LinkerOptions

        final_link = recorded.links[-1]
        assert final_link.cubin == result.metadata["cubin"]
        assert len(final_link.inputs) > 1
        linked = Linker(
            *final_link.inputs,
            options=LinkerOptions(arch="sm_80", link_time_optimization=True),
        ).link("cubin")
        assert linked.code

    assert recorded.nvrtc
    for request in recorded.nvrtc:
        assert request["cc"] == cc
        assert (
            f"--gpu-architecture=compute_{arch}".encode()
            in request["compiler_options"]
        )


@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_lineinfo_reaches_load_store_providers(
    monkeypatch: pytest.MonkeyPatch,
    items_per_thread,
) -> None:
    cuda, recorded = _record_compilations(monkeypatch)

    result = _compile_kernel(cuda, items_per_thread, lineinfo=True)

    assert result.metadata["cubin"]
    assert recorded.nvrtc
    assert all(
        b"-lineinfo" in request["compiler_options"]
        for request in recorded.nvrtc
    )


@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("lineinfo", [False, True])
def test_debug_kernel_links_generated_code_without_debug_information(
    monkeypatch: pytest.MonkeyPatch, items_per_thread, lineinfo
) -> None:
    # A debug kernel is linked with -g, and nvJitLink then rejects more than
    # one LTO input that carries debug or line information.
    cuda, recorded = _record_compilations(monkeypatch)

    result = _compile_kernel(
        cuda, items_per_thread, debug=True, opt=False, lineinfo=lineinfo
    )

    assert result.metadata["cubin"]
    assert recorded.nvrtc
    for request in recorded.nvrtc:
        assert b"-lineinfo" not in request["compiler_options"]
        assert b"-G" not in request["compiler_options"]
