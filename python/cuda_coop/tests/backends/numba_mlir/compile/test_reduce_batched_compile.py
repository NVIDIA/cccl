# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compile batched-reduction kernels and bundled native providers.

Run with GPUs hidden and target queries fixed at SM90, but keep real
compilation and linkage. Kernel cases check the inferred result shape and
reject a block group, an unknown output layout, and a partial physical warp.
A separate bundle compiles all numeric types across logical widths and
layouts without launching any provider.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
import numba_cuda_mlir.tools as compiler_tools
from numba_cuda_mlir import cuda, types

from cuda import coop
from cuda.coop.numba_mlir import _types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture
def compile_kernel(monkeypatch):
    """Provide real compilation with a fixed target and explicit launch facts.

    Require hidden GPUs and replace only target queries. The returned helper
    supplies float64 array arguments, an optional literal payload extent, and
    the chosen block size. Tests can inspect artifacts or planning failures
    without executing a kernel.
    """

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    device = SimpleNamespace(compute_capability=(9, 0))
    monkeypatch.setattr(_types.cuda, "get_current_device", lambda: device)
    monkeypatch.setattr(
        compiler_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )

    def compile_(kernel, block=64, items_per_thread=None):
        arguments = (types.float64[::1], types.float64[::1])
        if items_per_thread is not None:
            arguments += (types.IntegerLiteral(items_per_thread),)
        signature = types.void(*arguments)
        launch = (
            ("grid", (1, 1, 1)),
            ("block", (block, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        )
        return kernel._compile_launch_config_signature(signature, launch)

    return compile_


@pytest.mark.parametrize("layout", ["striped", "blocked"])
@pytest.mark.parametrize("items_per_thread", [1, 4, 33])
def test_reduce_batched_links_native_provider(
    compile_kernel, layout, items_per_thread
):
    """Link both result layouts across one- and two-slot output shapes.

    With a 32-lane warp, 1 and 4 batches need one result slot per lane, and
    33 batches need two. Read result.items_per_thread inside the kernel so
    compilation must propagate the computed extent before later indexing.
    Require a cubin and external link artifacts without executing this kernel.
    """

    @cuda.jit(chip="sm_90")
    def kernel(source, destination, items_per_thread):
        block = coop.this_block()
        values = coop.ThreadData(items_per_thread)
        coop.load(block, source, values)
        result = coop.reduce_batched(
            coop.this_warp(), values, output_layout=layout
        )
        output_items = result.items_per_thread
        for item in range(output_items):
            destination[cuda.threadIdx.x * output_items + item] = result[item]

    compiled = compile_kernel(kernel, items_per_thread=items_per_thread)
    assert compiled.metadata["cubin"]
    assert compiled.metadata["linked_external_link_items"]


@pytest.mark.parametrize(
    ("kind", "layout", "message"),
    [
        ("block", "striped", "group kind.*block"),
        ("warp", "broadcast", "output_layout"),
    ],
)
def test_reduce_batched_rejects_invalid_contract(
    compile_kernel, kind, layout, message
):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        values = coop.ThreadData(items_per_thread=3)
        coop.load(coop.this_block(), source, values)
        if kind == "block":
            group = coop.this_block()
        else:
            group = coop.this_warp()
        result = coop.reduce_batched(group, values, output_layout=layout)
        destination[cuda.threadIdx.x] = result[0]

    with pytest.raises(Exception, match=message):
        compile_kernel(kernel)


def test_reduce_batched_rejects_partial_physical_warp(compile_kernel):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        values = coop.ThreadData(items_per_thread=3)
        coop.load(coop.this_block(), source, values)
        result = coop.reduce_batched(coop.this_warp(), values)
        destination[cuda.threadIdx.x] = result[0]

    with pytest.raises(Exception, match="warp|multiple of 32"):
        compile_kernel(kernel, block=48)


def test_reduce_batched_provider_dtype_bundle(compile_kernel):
    """Compile numeric, width, and layout variants in one provider bundle.

    Collect each specialization separately, then give all of them the same
    NVRTC compiler context, because one bundle must use one compiler. Compile
    them into one link-time optimization (LTO) image, with one shared
    declaration of each payload type, including narrow integers and 64-bit
    types. Each provider keeps its logical width and block shape so that its
    scratch size is correct.
    """

    from cuda.coop.numba_mlir._compiler import _nvrtc
    from cuda.coop.numba_mlir._lowering._reduce_batched import reduce_batched

    context = _nvrtc.resolve_compile_context()
    collected = []
    for dtype_name in (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ):
        for width in (8, 32):
            for layout in ("blocked", "striped"):
                with _types.collect_specializations() as items:
                    reduce_batched(
                        getattr(types, dtype_name),
                        64,
                        width,
                        33,
                        output_layout=layout,
                    )
                assert len(items) == 1
                items[0]._compile_context = context
                collected.extend(items)
    ltoir = _types.prepare_ltoir_bundle(
        collected,
        allow_single=True,
    )
    assert isinstance(ltoir, bytes) and ltoir
