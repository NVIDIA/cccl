# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Link group helpers and Reduce callbacks for explicit kernel targets."""

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import cuda, types

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_LAUNCH_CONFIG_KEY = (
    ("grid", (1, 1, 1)),
    ("block", (64, 1, 1)),
    ("sharedmem", 0),
    ("cluster", None),
)


@pytest.fixture
def recorded(monkeypatch):
    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir.linker import Linker

    from cuda.coop.numba_mlir._compiler import _nvrtc

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )
    calls = SimpleNamespace(nvrtc=[], callbacks=[], links=[])
    compile_impl = _nvrtc.compile_impl
    compile_callback = cuda.compile
    complete = Linker.complete

    def record_nvrtc(*args, **kwargs):
        calls.nvrtc.append(kwargs)
        return compile_impl(*args, **kwargs)

    def record_callback(*args, **kwargs):
        calls.callbacks.append(kwargs)
        return compile_callback(*args, **kwargs)

    def record_link(self):
        result = complete(self)
        calls.links.append(tuple(self._object_codes))
        return result

    monkeypatch.setattr(_nvrtc, "compile_impl", record_nvrtc)
    monkeypatch.setattr(cuda, "compile", record_callback)
    monkeypatch.setattr(Linker, "complete", record_link)
    # Keep the callback cache intact across every compilation in each test.
    return calls


def _compile_kernel(callback, marker=0, **jit_options):
    import cuda.coop.numba_mlir as coop

    @cuda.jit(**jit_options)
    def kernel(source, output, ranks):
        block = coop.this_block()
        rank = block.rank()
        storage = coop.TempStorage(sharing="shared", auto_sync=True)
        value = coop.reduce(
            block, source[rank], binary_op=callback, temp_storage=storage
        )
        if rank == 0:
            output[0] = value
        ranks[rank] = rank + marker

    return kernel._compile_launch_config_signature(
        types.void(types.int32[::1], types.int32[::1], types.uint32[::1]),
        _LAUNCH_CONFIG_KEY,
    )


def test_reduce_callback_cache_preserves_requested_targets(recorded):
    from cuda.core import Linker, LinkerOptions

    @cuda.jit(device=True)
    def maximum(left, right):
        return left if left > right else right  # noqa: FURB136

    for marker, cc in enumerate(((8, 0), (8, 6), (8, 0))):
        start = len(recorded.nvrtc)
        links_before = len(recorded.links)
        result = _compile_kernel(maximum, marker, cc=cc)
        assert result.metadata["cubin"]
        assert len(recorded.links) > links_before
        if cc == (8, 0):
            linked = Linker(
                *recorded.links[-1],
                options=LinkerOptions(
                    arch="sm_80", link_time_optimization=True
                ),
            ).link("cubin")
            assert linked.code
        requests = recorded.nvrtc[start:]
        assert requests
        assert all(request["cc"] == cc[0] * 10 + cc[1] for request in requests)
        assert any(
            "cuda_coop_numba_mlir_group_" in request["cpp"]
            for request in requests
        )
        assert any("cub::BlockReduce" in request["cpp"] for request in requests)
        expected = [(8, 0)] if marker == 0 else [(8, 0), (8, 6)]
        assert [request["cc"] for request in recorded.callbacks] == expected


def test_reduce_lineinfo_reaches_helpers_and_providers(recorded):
    @cuda.jit(device=True)
    def add(left, right):
        return left + right

    result = _compile_kernel(add, lineinfo=True)
    assert result.metadata["cubin"]
    assert recorded.nvrtc
    assert all(
        b"-lineinfo" in request["compiler_options"]
        for request in recorded.nvrtc
    )
    assert [request["cc"] for request in recorded.callbacks] == [(9, 0)]
    assert all(
        "lineinfo" not in request and "arch_suffix" not in request
        for request in recorded.callbacks
    )


@pytest.mark.parametrize(
    "with_primitives,bundle_failure",
    [(False, False), (True, False), (True, True)],
    ids=["queries", "mixed", "fallback"],
)
def test_group_queries_share_provider_compilation(
    recorded, monkeypatch, with_primitives, bundle_failure
):
    """Compile the walkthrough's queries with optional Load/Store providers.

    Distinct query and synchronization helpers must share one provider image,
    including when the kernel has no CUB primitive to initiate collection.
    """

    from cuda import coop

    if bundle_failure:
        from cuda.coop.numba_mlir._compiler import _rewrite_invocables

        def fail_bundle(*args, **kwargs):
            raise RuntimeError("exercise individual provider fallback")

        monkeypatch.setattr(
            _rewrite_invocables, "prepare_ltoir_bundle", fail_bundle
        )

    @cuda.jit(chip="sm_80", lineinfo=True)
    def kernel(source, output, queries):
        thread = coop.this_thread()
        warp = coop.this_warp()
        block = coop.this_block()
        grid = coop.this_grid()
        lanes = warp.group_by(8)
        index = cuda.grid(1)

        thread.sync()
        warp.sync_aligned()
        lanes.sync_aligned()
        block.sync()
        queries[0, index] = thread.rank("block")
        queries[1, index] = warp.rank()
        queries[2, index] = block.rank_as(types.int64, "grid")
        queries[3, index] = block.count_as(types.int64)
        queries[4, index] = grid.rank()
        queries[5, index] = grid.count()
        queries[6, index] = lanes.rank()
        queries[7, index] = lanes.count()
        if with_primitives:
            values = coop.ThreadData(items_per_thread=4)
            coop.load(block, source, values, algorithm="transpose")
            coop.store(block, output, values, algorithm="transpose")

    result = kernel._compile_launch_config_signature(
        types.void(types.int32[::1], types.int32[::1], types.int64[:, ::1]),
        _LAUNCH_CONFIG_KEY,
    )

    assert result.metadata["cubin"]
    assert len(recorded.nvrtc) == (14 if bundle_failure else 1)
    sources = "\n".join(request["cpp"] for request in recorded.nvrtc)
    if not bundle_failure:
        includes = [
            line
            for line in sources.splitlines()
            if line.startswith("#include ")
        ]
        assert len(includes) == len(set(includes))
    assert "cuda_coop_numba_mlir_group_" in sources
    assert ("cub::BlockLoad" in sources) == with_primitives
    assert ("cub::BlockStore" in sources) == with_primitives
    assert all(request["cc"] == 80 for request in recorded.nvrtc)
    assert all(
        b"-lineinfo" in request["compiler_options"]
        for request in recorded.nvrtc
    )
