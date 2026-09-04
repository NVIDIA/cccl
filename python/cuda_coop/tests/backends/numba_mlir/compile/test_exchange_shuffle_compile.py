# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free, real-toolchain compilation contracts for Exchange and Shuffle."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cuda.coop._core import ArgumentBinding, SynchronizationScope
from cuda.coop.numba_mlir import _types
from cuda.coop.numba_mlir._compiler import _nvrtc
from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._lowering import _exchange, _shuffle
from cuda.coop.numba_mlir._types import algo_coalesce_key

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_FIXED_COMPUTE_CAPABILITY = (9, 0)
_BLOCK_THREADS = 64
_ITEMS_PER_THREAD = 2
_BLOCK_MODES = (
    "striped_to_blocked",
    "blocked_to_striped",
    "warp_striped_to_blocked",
    "blocked_to_warp_striped",
    "scatter_to_blocked",
    "scatter_to_striped",
    "scatter_to_striped_guarded",
    "scatter_to_striped_flagged",
)
_WARP_MODES = (
    "striped_to_blocked",
    "blocked_to_striped",
    "scatter_to_striped",
)
_LOGICAL_WARP_WIDTHS = (1, 2, 4, 8, 16, 32)
_DTYPES = (
    types.int8,
    types.uint8,
    types.int16,
    types.uint16,
    types.int32,
    types.uint32,
    types.int64,
    types.uint64,
    types.float32,
    types.float64,
)


@pytest.fixture(autouse=True)
def _fixed_current_device(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Hide runtime discovery while leaving NVRTC and nvJitLink real."""

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "the Numba-CUDA-MLIR compile stage must hide all CUDA devices"
    )
    queries: list[tuple[int, int]] = []

    def current_device() -> SimpleNamespace:
        queries.append(_FIXED_COMPUTE_CAPABILITY)
        return SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    monkeypatch.setattr(_types.cuda, "get_current_device", current_device)
    return queries


@pytest.fixture(scope="module")
def compile_context() -> _nvrtc.CompileContext:
    return _nvrtc.resolve_compile_context()


def _collect(factory, compile_context: _nvrtc.CompileContext, /, **kwargs):
    with _types.collect_specializations() as collected:
        factory(**kwargs)
    assert len(collected) == 1
    algorithm, threads, block_threads = collected[0]
    algorithm._compile_context = compile_context
    return algorithm, threads, block_threads


def _source(collected) -> str:
    algorithm, threads, block_threads = collected
    return algorithm._source_code(
        threads=threads,
        block_threads=block_threads,
    )[0]


def _compile_bundle(
    collected,
    *,
    bundle_name: str,
) -> bytes:
    algorithms = [item[0] for item in collected]
    ltoir = _types.prepare_ltoir_bundle(
        algorithms,
        bundle_name=bundle_name,
        allow_single=True,
        threads_by_algo={id(item[0]): item[1] for item in collected},
        block_threads_by_algo={id(item[0]): item[2] for item in collected},
    )
    assert isinstance(ltoir, bytes)
    assert ltoir
    return ltoir


def _block_exchange(
    compile_context: _nvrtc.CompileContext,
    *,
    mode: str,
    dtype=types.int32,
    warp_time_slicing: bool = False,
):
    kwargs = {
        "dtype": dtype,
        "threads_per_block": _BLOCK_THREADS,
        "items_per_thread": _ITEMS_PER_THREAD,
        "mode": mode,
        "warp_time_slicing": warp_time_slicing,
    }
    if mode == "scatter_to_striped_flagged":
        factory = _exchange.exchange_flagged
        kwargs.update(rank_dtype=types.int32, valid_flag_dtype=types.uint8)
    elif mode.startswith("scatter_to_"):
        factory = _exchange.exchange_ranked
        kwargs["rank_dtype"] = types.int32
    else:
        factory = _exchange.exchange
    return _collect(factory, compile_context, **kwargs)


def _warp_exchange(
    compile_context: _nvrtc.CompileContext,
    *,
    mode: str,
    threads_in_warp: int,
    dtype=types.int32,
):
    kwargs = {
        "dtype": dtype,
        "threads_per_block": _BLOCK_THREADS,
        "threads_in_warp": threads_in_warp,
        "items_per_thread": _ITEMS_PER_THREAD,
        "mode": mode,
    }
    if mode == "scatter_to_striped":
        factory = _exchange.warp_exchange_ranked
        kwargs["rank_dtype"] = types.int32
    else:
        factory = _exchange.warp_exchange
    return _collect(factory, compile_context, **kwargs)


def test_all_block_exchange_modes_compile_with_owned_storage(
    compile_context: _nvrtc.CompileContext,
    _fixed_current_device: list[tuple[int, int]],
) -> None:
    collected = [_block_exchange(compile_context, mode=mode) for mode in _BLOCK_MODES]

    for mode, item in zip(_BLOCK_MODES, collected):
        algorithm = item[0]
        source = _source(item)
        method = "".join(part.title() for part in mode.split("_"))
        assert f".{method}(" in source
        assert "cub::BlockExchange<" in source
        assert "TempStorage" in source
        assert "__shared__" in source
        assert "__syncthreads();" in source
        assert "__syncwarp" not in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.BLOCK
        assert algorithm.synchronization_scope is SynchronizationScope.BLOCK

    _compile_bundle(
        collected,
        bundle_name="cuda_coop_numba_mlir_all_block_exchange_modes",
    )
    assert all(item[0].temp_storage_bytes > 0 for item in collected)
    assert all(item[0].temp_storage_alignment > 0 for item in collected)
    assert _fixed_current_device


def test_block_exchange_time_slicing_changes_storage_identity_and_size(
    compile_context: _nvrtc.CompileContext,
) -> None:
    ordinary = _block_exchange(
        compile_context,
        mode="blocked_to_warp_striped",
    )
    sliced = _block_exchange(
        compile_context,
        mode="blocked_to_warp_striped",
        warp_time_slicing=True,
    )
    ordinary_source = _source(ordinary)
    sliced_source = _source(sliced)
    assert "cub::BlockExchange<::cuda::std::int32_t, 64, 2, 0, 1, 1>" in (
        ordinary_source
    )
    assert "cub::BlockExchange<::cuda::std::int32_t, 64, 2, 1, 1, 1>" in (sliced_source)
    assert algo_coalesce_key(ordinary[0]) != algo_coalesce_key(sliced[0])

    _compile_bundle(
        [ordinary, sliced],
        bundle_name="cuda_coop_numba_mlir_block_exchange_time_slicing",
    )
    assert 0 < sliced[0].temp_storage_bytes < ordinary[0].temp_storage_bytes


def test_all_warp_exchange_modes_and_logical_widths_compile(
    compile_context: _nvrtc.CompileContext,
) -> None:
    collected = [
        _warp_exchange(
            compile_context,
            mode=mode,
            threads_in_warp=width,
        )
        for width in _LOGICAL_WARP_WIDTHS
        for mode in _WARP_MODES
    ]

    for item, (width, mode) in zip(
        collected,
        ((width, mode) for width in _LOGICAL_WARP_WIDTHS for mode in _WARP_MODES),
    ):
        algorithm = item[0]
        source = _source(item)
        method = "".join(part.title() for part in mode.split("_"))
        assert f".{method}(" in source
        assert (
            "cub::WarpExchange<::cuda::std::int32_t, 2, "
            f"{width}, ::cub::WARP_EXCHANGE_SMEM>"
        ) in source
        assert f"temp_storages[{_BLOCK_THREADS // width}]" in source
        assert "__syncthreads" not in source
        if width == 32:
            assert "__syncwarp();" in source
        else:
            assert (
                "__syncwarp((((1u << "
                f"{width}) - 1u) << (((__coop_thread_rank & 31) / "
                f"{width}) * {width})));"
            ) in source
            assert "__syncwarp();" not in source
        assert algorithm.storage_abi is StorageABI.LEADING_POINTER
        assert algorithm.execution_scope is SynchronizationScope.WARP
        assert algorithm.synchronization_scope is SynchronizationScope.WARP

    _compile_bundle(
        collected,
        bundle_name="cuda_coop_numba_mlir_all_warp_exchange_widths",
    )
    assert all(item[0].temp_storage_bytes > 0 for item in collected)
    assert len({algo_coalesce_key(item[0]) for item in collected}) == len(collected)


def test_shuffle_modes_compile_with_exact_distance_abis(
    compile_context: _nvrtc.CompileContext,
) -> None:
    static = [
        _collect(
            _shuffle.shuffle_scalar,
            compile_context,
            dtype=types.int32,
            threads_per_block=_BLOCK_THREADS,
            mode="offset",
            distance=ArgumentBinding.static(-3),
        ),
        _collect(
            _shuffle.shuffle_scalar,
            compile_context,
            dtype=types.int32,
            threads_per_block=_BLOCK_THREADS,
            mode="rotate",
            distance=ArgumentBinding.static(3),
        ),
        _collect(
            _shuffle.shuffle_array,
            compile_context,
            dtype=types.int32,
            threads_per_block=_BLOCK_THREADS,
            items_per_thread=_ITEMS_PER_THREAD,
            mode="up",
        ),
        _collect(
            _shuffle.shuffle_array,
            compile_context,
            dtype=types.int32,
            threads_per_block=_BLOCK_THREADS,
            items_per_thread=_ITEMS_PER_THREAD,
            mode="down",
        ),
    ]
    runtime_offset = _collect(
        _shuffle.shuffle_scalar,
        compile_context,
        dtype=types.int32,
        threads_per_block=_BLOCK_THREADS,
        mode="offset",
        distance=ArgumentBinding.runtime(),
    )
    runtime_rotate = _collect(
        _shuffle.shuffle_scalar,
        compile_context,
        dtype=types.int32,
        threads_per_block=_BLOCK_THREADS,
        mode="rotate",
        distance=ArgumentBinding.runtime(),
    )

    sources = [_source(item) for item in static]
    assert ".Offset(param_0, param_1, -3)" in sources[0]
    assert ".Rotate(param_0, param_1, 3)" in sources[1]
    assert ".Up(" in sources[2]
    assert ".Down(" in sources[3]
    for source in sources:
        assert "cub::BlockShuffle<" in source
        assert "__shared__" in source
        assert "__syncthreads();" in source
        assert "__syncwarp" not in source

    offset_source = _source(runtime_offset)
    rotate_source = _source(runtime_rotate)
    assert "::cuda::std::int64_t param_2" in offset_source
    assert "param_2 < -2147483648 || param_2 > 2147483647" in offset_source
    assert "static_cast<::cuda::std::int32_t>(param_2)" in offset_source
    assert "::cuda::std::int64_t param_2" in rotate_source
    assert "param_2 < 1 || param_2 > 63" in rotate_source
    assert "static_cast<::cuda::std::uint32_t>(param_2)" in rotate_source
    assert 'asm volatile("trap;" : : :);' in offset_source
    assert 'asm volatile("trap;" : : :);' in rotate_source

    all_items = [*static, runtime_offset, runtime_rotate]
    _compile_bundle(
        all_items,
        bundle_name="cuda_coop_numba_mlir_all_block_shuffle_modes",
    )
    assert all(item[0].temp_storage_bytes > 0 for item in all_items)


def test_exchange_and_shuffle_compile_for_every_supported_dtype(
    compile_context: _nvrtc.CompileContext,
) -> None:
    collected = []
    for dtype in _DTYPES:
        collected.extend(
            (
                _block_exchange(
                    compile_context,
                    mode="striped_to_blocked",
                    dtype=dtype,
                ),
                _warp_exchange(
                    compile_context,
                    mode="blocked_to_striped",
                    threads_in_warp=8,
                    dtype=dtype,
                ),
                _collect(
                    _shuffle.shuffle_array,
                    compile_context,
                    dtype=dtype,
                    threads_per_block=_BLOCK_THREADS,
                    items_per_thread=_ITEMS_PER_THREAD,
                    mode="down",
                ),
                _collect(
                    _shuffle.shuffle_scalar,
                    compile_context,
                    dtype=dtype,
                    threads_per_block=_BLOCK_THREADS,
                    mode="offset",
                    distance=ArgumentBinding.static(-3),
                ),
                _collect(
                    _shuffle.shuffle_scalar,
                    compile_context,
                    dtype=dtype,
                    threads_per_block=_BLOCK_THREADS,
                    mode="rotate",
                    distance=ArgumentBinding.static(3),
                ),
            )
        )

    _compile_bundle(
        collected,
        bundle_name="cuda_coop_numba_mlir_exchange_shuffle_dtypes",
    )
    assert all(item[0].temp_storage_bytes > 0 for item in collected)


def _production_compile_environment(monkeypatch: pytest.MonkeyPatch):
    import numba_cuda_mlir.tools as numba_mlir_tools
    from numba_cuda_mlir import cuda as compiler_cuda

    fixed_device = SimpleNamespace(compute_capability=_FIXED_COMPUTE_CAPABILITY)

    def fixed_compute_capability(as_type=str):
        assert as_type in (str, tuple)
        return _FIXED_COMPUTE_CAPABILITY if as_type is tuple else "sm_90"

    monkeypatch.setattr(
        numba_mlir_tools,
        "get_gpu_compute_capability",
        fixed_compute_capability,
    )
    monkeypatch.setattr(compiler_cuda, "get_current_device", lambda: fixed_device)
    return compiler_cuda


def test_production_kernel_compile_links_shared_storage_and_barriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler_cuda = _production_compile_environment(monkeypatch)

    import cuda.coop.numba_mlir as qualified_coop
    from cuda import coop as portable_coop

    @compiler_cuda.jit(chip="sm_90")
    def kernel(source, destination, distance):
        thread = compiler_cuda.threadIdx.x
        payload = qualified_coop.ThreadData(2, dtype=types.int32)
        payload[0] = source[thread * 2]
        payload[1] = source[thread * 2 + 1]
        exchanged = portable_coop.exchange(
            portable_coop.this_block(),
            payload,
            mode="blocked_to_striped",
        )
        shifted = portable_coop.shuffle(
            portable_coop.this_block(),
            exchanged,
            mode="up",
        )
        rotated = qualified_coop.shuffle(
            qualified_coop.this_block(),
            source[thread],
            mode="rotate",
            distance=distance,
        )
        destination[thread * 2] = shifted[0] + rotated
        destination[thread * 2 + 1] = shifted[1]

    signature = types.void(types.int32[::1], types.int32[::1], types.int32)
    launch_config_key = (
        ("grid", (1, 1, 1)),
        ("block", (_BLOCK_THREADS, 1, 1)),
        ("sharedmem", 0),
        ("cluster", None),
    )
    result = kernel._compile_launch_config_signature(
        signature,
        launch_config_key,
    )
    assert isinstance(result.metadata["ltoir"], bytes)
    assert result.metadata["ltoir"]
    assert isinstance(result.metadata["cubin"], bytes)
    assert result.metadata["cubin"]
    assert result.metadata["linked_external_link_items"]

    ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert ".visible .entry" in ptx
    assert ".shared" in ptx
    assert "bar.sync" in ptx
