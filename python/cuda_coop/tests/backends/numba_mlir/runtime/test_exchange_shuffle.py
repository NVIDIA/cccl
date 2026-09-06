# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Exchange and Shuffle runtime qualification for Numba-CUDA-MLIR."""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as qualified_coop
from cuda import coop as root_coop

assert qualified_coop.__file__ is not None
_QUALIFIED_COOP_ORIGIN = Path(qualified_coop.__file__).resolve()
_SAFE_PATH_FLAG = "-P" if sys.version_info >= (3, 11) else "-I"

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]

_BLOCK_THREADS = 64
_WARP_THREADS = 32
_LOGICAL_WARP_THREADS = 8
_ITEMS_PER_THREAD = 3
_TILE_ITEMS = _BLOCK_THREADS * _ITEMS_PER_THREAD
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
_TIME_SLICED_MODES = tuple(
    mode
    for mode in _BLOCK_MODES
    if mode not in {"scatter_to_striped_guarded", "scatter_to_striped_flagged"}
)


def _values(size: int, *, shift: int = 0) -> np.ndarray:
    return ((np.arange(size, dtype=np.int64) * 17 + shift) % 997 - 491).astype(np.int32)


def _structured_exchange_oracle(
    source: np.ndarray,
    *,
    group_width: int,
    items_per_thread: int,
    mode: str,
) -> np.ndarray:
    """Compute a layout transform without applying its inverse operation."""

    result = np.empty_like(source)
    group_items = group_width * items_per_thread
    group_count = source.size // group_items
    for group_index in range(group_count):
        first_thread = group_index * group_width
        for lane in range(group_width):
            for item in range(items_per_thread):
                output_index = (first_thread + lane) * items_per_thread + item
                if mode == "striped_to_blocked":
                    shared_index = lane * items_per_thread + item
                    input_lane = shared_index % group_width
                    input_item = shared_index // group_width
                elif mode == "blocked_to_striped":
                    shared_index = item * group_width + lane
                    input_lane = shared_index // items_per_thread
                    input_item = shared_index % items_per_thread
                else:
                    raise AssertionError(f"unexpected structured mode {mode!r}")
                input_index = (
                    first_thread + input_lane
                ) * items_per_thread + input_item
                result[output_index] = source[input_index]
    return result


def _warp_structured_exchange_oracle(
    source: np.ndarray,
    *,
    items_per_thread: int,
    mode: str,
) -> np.ndarray:
    mapped_mode = {
        "warp_striped_to_blocked": "striped_to_blocked",
        "blocked_to_warp_striped": "blocked_to_striped",
    }[mode]
    return _structured_exchange_oracle(
        source,
        group_width=_WARP_THREADS,
        items_per_thread=items_per_thread,
        mode=mapped_mode,
    )


def _reversed_ranks(
    *,
    thread_count: int,
    group_width: int,
    items_per_thread: int,
) -> np.ndarray:
    ranks = np.empty(thread_count * items_per_thread, dtype=np.int32)
    group_items = group_width * items_per_thread
    for thread in range(thread_count):
        lane = thread % group_width
        for item in range(items_per_thread):
            local_index = lane * items_per_thread + item
            ranks[thread * items_per_thread + item] = group_items - 1 - local_index
    return ranks


def _scatter_exchange_oracle(
    source: np.ndarray,
    ranks: np.ndarray,
    *,
    group_width: int,
    items_per_thread: int,
    mode: str,
    valid_flags: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return values and a mask for destinations written by active inputs."""

    result = np.empty_like(source)
    compared = np.zeros(source.size, dtype=np.bool_)
    group_items = group_width * items_per_thread
    group_count = source.size // group_items
    for group_index in range(group_count):
        first_thread = group_index * group_width
        values_by_rank: dict[int, np.int32] = {}
        for lane in range(group_width):
            for item in range(items_per_thread):
                input_index = (first_thread + lane) * items_per_thread + item
                rank = int(ranks[input_index])
                active = rank >= 0
                if valid_flags is not None:
                    active = active and bool(valid_flags[input_index])
                if active:
                    values_by_rank[rank] = source[input_index]

        for lane in range(group_width):
            for item in range(items_per_thread):
                output_index = (first_thread + lane) * items_per_thread + item
                if mode == "scatter_to_blocked":
                    output_rank = lane * items_per_thread + item
                else:
                    output_rank = item * group_width + lane
                if output_rank in values_by_rank:
                    result[output_index] = values_by_rank[output_rank]
                    compared[output_index] = True
    return result, compared


@lru_cache(maxsize=None)
def _structured_exchange_kernel(
    scope: str,
    mode: str,
    qualified: bool,
):
    if scope == "block" and qualified:

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.exchange(
                qualified_coop.this_block(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif scope == "block":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = root_coop.exchange(
                root_coop.this_block(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif scope == "warp" and qualified:

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.exchange(
                qualified_coop.this_warp(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif scope == "warp":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = root_coop.exchange(
                root_coop.this_warp(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif scope == "logical" and qualified:

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.exchange(
                qualified_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif scope == "logical":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = root_coop.exchange(
                root_coop.this_warp().group_by(_LOGICAL_WARP_THREADS),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    else:
        raise AssertionError(f"unexpected Exchange scope {scope!r}")

    return kernel


@pytest.mark.parametrize("qualified", (False, True), ids=("portable", "qualified"))
@pytest.mark.parametrize("mode", ("striped_to_blocked", "blocked_to_striped"))
@pytest.mark.parametrize(
    ("scope", "group_width"),
    (
        ("block", _BLOCK_THREADS),
        ("warp", _WARP_THREADS),
        ("logical", _LOGICAL_WARP_THREADS),
    ),
    ids=("block", "physical-warp", "logical-warp"),
)
def test_common_exchange_layouts_match_independent_oracles_and_preserve_input(
    qualified: bool,
    mode: str,
    scope: str,
    group_width: int,
) -> None:
    source = _values(_TILE_ITEMS, shift=23)
    observed = np.full(_TILE_ITEMS, -2001, dtype=np.int32)
    preserved = np.full(_TILE_ITEMS, -2003, dtype=np.int32)
    expected = _structured_exchange_oracle(
        source,
        group_width=group_width,
        items_per_thread=_ITEMS_PER_THREAD,
        mode=mode,
    )

    _structured_exchange_kernel(scope, mode, qualified)[1, _BLOCK_THREADS](
        source,
        observed,
        preserved,
    )

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _untyped_load_exchange_kernel(source, observed):
    thread = cuda.threadIdx.x
    payload = qualified_coop.ThreadData(2)
    loaded = root_coop.load(
        root_coop.this_block(),
        source,
        payload,
        algorithm="direct",
    )
    exchanged = root_coop.exchange(
        root_coop.this_block(),
        loaded,
        mode="blocked_to_striped",
    )
    observed[thread * 2] = exchanged[0]
    observed[thread * 2 + 1] = exchanged[1]


def test_untyped_load_result_composes_directly_into_exchange() -> None:
    items_per_thread = 2
    source = _values(_BLOCK_THREADS * items_per_thread, shift=29)
    observed = np.full(source.size, -2005, dtype=np.int32)
    expected = _structured_exchange_oracle(
        source,
        group_width=_BLOCK_THREADS,
        items_per_thread=items_per_thread,
        mode="blocked_to_striped",
    )

    _untyped_load_exchange_kernel[1, _BLOCK_THREADS](source, observed)

    np.testing.assert_array_equal(observed, expected)


@lru_cache(maxsize=None)
def _qualified_block_exchange_kernel(mode: str, warp_time_slicing: bool):
    if mode in {
        "striped_to_blocked",
        "blocked_to_striped",
        "warp_striped_to_blocked",
        "blocked_to_warp_striped",
    }:

        @cuda.jit
        def kernel(source, ranks_source, flags_source, observed, preserved, ranks_out):
            thread = cuda.threadIdx.x
            payload = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.exchange(
                qualified_coop.this_block(),
                payload,
                mode=mode,
                warp_time_slicing=warp_time_slicing,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif mode == "scatter_to_striped_flagged":

        @cuda.jit
        def kernel(source, ranks_source, flags_source, observed, preserved, ranks_out):
            thread = cuda.threadIdx.x
            payload = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            ranks = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            valid_flags = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int8,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                payload[item] = source[index]
                ranks[item] = ranks_source[index]
                valid_flags[item] = flags_source[index]
            result = qualified_coop.exchange(
                qualified_coop.this_block(),
                payload,
                mode=mode,
                ranks=ranks,
                valid_flags=valid_flags,
                warp_time_slicing=warp_time_slicing,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]
                ranks_out[index] = ranks[item]

    else:

        @cuda.jit
        def kernel(source, ranks_source, flags_source, observed, preserved, ranks_out):
            thread = cuda.threadIdx.x
            payload = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            ranks = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                payload[item] = source[index]
                ranks[item] = ranks_source[index]
            result = qualified_coop.exchange(
                qualified_coop.this_block(),
                payload,
                mode=mode,
                ranks=ranks,
                warp_time_slicing=warp_time_slicing,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]
                ranks_out[index] = ranks[item]

    return kernel


def _block_exchange_inputs(
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = _values(_TILE_ITEMS, shift=71)
    ranks = _reversed_ranks(
        thread_count=_BLOCK_THREADS,
        group_width=_BLOCK_THREADS,
        items_per_thread=_ITEMS_PER_THREAD,
    )
    valid_flags = np.ones(_TILE_ITEMS, dtype=np.int8)
    if mode == "scatter_to_striped_guarded":
        ranks[::7] = -1
    elif mode == "scatter_to_striped_flagged":
        valid_flags[::7] = 0

    if mode in {"striped_to_blocked", "blocked_to_striped"}:
        expected = _structured_exchange_oracle(
            source,
            group_width=_BLOCK_THREADS,
            items_per_thread=_ITEMS_PER_THREAD,
            mode=mode,
        )
        compared = np.ones(_TILE_ITEMS, dtype=np.bool_)
    elif mode in {"warp_striped_to_blocked", "blocked_to_warp_striped"}:
        expected = _warp_structured_exchange_oracle(
            source,
            items_per_thread=_ITEMS_PER_THREAD,
            mode=mode,
        )
        compared = np.ones(_TILE_ITEMS, dtype=np.bool_)
    else:
        expected, compared = _scatter_exchange_oracle(
            source,
            ranks,
            group_width=_BLOCK_THREADS,
            items_per_thread=_ITEMS_PER_THREAD,
            mode=mode,
            valid_flags=(valid_flags if mode == "scatter_to_striped_flagged" else None),
        )
    return source, ranks, valid_flags, expected, compared


@pytest.mark.parametrize("mode", _BLOCK_MODES)
def test_qualified_block_exchange_modes_match_oracles_and_accept_local_arrays(
    mode: str,
) -> None:
    source, ranks, valid_flags, expected, compared = _block_exchange_inputs(mode)
    observed = np.full(_TILE_ITEMS, -2011, dtype=np.int32)
    preserved = np.full(_TILE_ITEMS, -2013, dtype=np.int32)
    ranks_out = np.full(_TILE_ITEMS, -2017, dtype=np.int32)

    _qualified_block_exchange_kernel(mode, False)[1, _BLOCK_THREADS](
        source,
        ranks,
        valid_flags,
        observed,
        preserved,
        ranks_out,
    )

    np.testing.assert_array_equal(observed[compared], expected[compared])
    np.testing.assert_array_equal(preserved, source)
    if mode.startswith("scatter_"):
        np.testing.assert_array_equal(ranks_out, ranks)


@pytest.mark.parametrize("mode", _TIME_SLICED_MODES)
def test_block_exchange_warp_time_slicing_matches_the_full_storage_oracle(
    mode: str,
) -> None:
    source, ranks, valid_flags, expected, compared = _block_exchange_inputs(mode)
    observed = np.full(_TILE_ITEMS, -2021, dtype=np.int32)
    preserved = np.full(_TILE_ITEMS, -2023, dtype=np.int32)
    ranks_out = np.full(_TILE_ITEMS, -2027, dtype=np.int32)

    _qualified_block_exchange_kernel(mode, True)[1, _BLOCK_THREADS](
        source,
        ranks,
        valid_flags,
        observed,
        preserved,
        ranks_out,
    )

    np.testing.assert_array_equal(observed[compared], expected[compared])
    np.testing.assert_array_equal(preserved, source)
    if mode.startswith("scatter_"):
        np.testing.assert_array_equal(ranks_out, ranks)


@lru_cache(maxsize=None)
def _repeated_warp_exchange_kernel(width: int):
    if width == _WARP_THREADS:

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            striped = qualified_coop.exchange(
                qualified_coop.this_warp(),
                payload,
                mode="blocked_to_striped",
            )
            blocked = qualified_coop.exchange(
                qualified_coop.this_warp(),
                striped,
                mode="striped_to_blocked",
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = blocked[item]

    else:

        @cuda.jit
        def kernel(source, observed):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            striped = qualified_coop.exchange(
                qualified_coop.this_warp().group_by(width),
                payload,
                mode="blocked_to_striped",
            )
            blocked = qualified_coop.exchange(
                qualified_coop.this_warp().group_by(width),
                striped,
                mode="striped_to_blocked",
            )
            for item in range(_ITEMS_PER_THREAD):
                observed[thread * _ITEMS_PER_THREAD + item] = blocked[item]

    return kernel


@pytest.mark.parametrize(
    "width",
    (
        pytest.param(_WARP_THREADS, id="physical"),
        pytest.param(_LOGICAL_WARP_THREADS, id="logical"),
    ),
)
def test_repeated_warp_exchange_reuses_isolated_group_storage(width: int) -> None:
    source = _values(_TILE_ITEMS, shift=139)
    observed = np.full(_TILE_ITEMS, -2049, dtype=np.int32)

    _repeated_warp_exchange_kernel(width)[1, _BLOCK_THREADS](source, observed)

    np.testing.assert_array_equal(observed, source)


@lru_cache(maxsize=None)
def _array_shuffle_kernel(mode: str, api: str):
    if api == "portable":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = root_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = root_coop.shuffle(
                root_coop.this_block(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif api == "qualified-thread-data":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = qualified_coop.ThreadData(
                _ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.shuffle(
                qualified_coop.this_block(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    elif api == "qualified-local-array":

        @cuda.jit
        def kernel(source, observed, preserved):
            thread = cuda.threadIdx.x
            payload = cuda.local.array(
                shape=_ITEMS_PER_THREAD,
                dtype=types.int32,
            )
            for item in range(_ITEMS_PER_THREAD):
                payload[item] = source[thread * _ITEMS_PER_THREAD + item]
            result = qualified_coop.shuffle(
                qualified_coop.this_block(),
                payload,
                mode=mode,
            )
            for item in range(_ITEMS_PER_THREAD):
                index = thread * _ITEMS_PER_THREAD + item
                observed[index] = result[item]
                preserved[index] = payload[item]

    else:
        raise AssertionError(f"unexpected Shuffle API {api!r}")

    return kernel


@pytest.mark.parametrize(
    "api",
    ("portable", "qualified-thread-data", "qualified-local-array"),
)
@pytest.mark.parametrize("mode", ("up", "down"))
def test_array_shuffle_matches_a_flattened_oracle_and_preserves_input(
    api: str,
    mode: str,
) -> None:
    source = _values(_TILE_ITEMS, shift=149)
    observed = np.full(_TILE_ITEMS, -2051, dtype=np.int32)
    preserved = np.full(_TILE_ITEMS, -2053, dtype=np.int32)

    _array_shuffle_kernel(mode, api)[1, _BLOCK_THREADS](
        source,
        observed,
        preserved,
    )

    if mode == "up":
        np.testing.assert_array_equal(observed[1:], source[:-1])
    else:
        np.testing.assert_array_equal(observed[:-1], source[1:])
    np.testing.assert_array_equal(preserved, source)


@cuda.jit
def _offset_shuffle(source, distances, observed):
    thread = cuda.threadIdx.x
    observed[thread] = qualified_coop.shuffle(
        qualified_coop.this_block(),
        source[thread],
        mode="offset",
        distance=distances[thread],
    )


def test_scalar_offset_accepts_per_thread_negative_and_positive_int32_distances() -> (
    None
):
    source = _values(_BLOCK_THREADS, shift=173)
    distances = ((np.arange(_BLOCK_THREADS) % 9) - 4).astype(np.int32)
    observed = np.full(_BLOCK_THREADS, -2061, dtype=np.int32)
    compared = np.zeros(_BLOCK_THREADS, dtype=np.bool_)
    expected = np.empty_like(source)
    for thread, distance in enumerate(distances):
        source_thread = thread + int(distance)
        if 0 <= source_thread < _BLOCK_THREADS:
            expected[thread] = source[source_thread]
            compared[thread] = True

    _offset_shuffle[1, _BLOCK_THREADS](source, distances, observed)

    assert np.any(distances < 0)
    assert np.any(distances > 0)
    np.testing.assert_array_equal(observed[compared], expected[compared])


@lru_cache(maxsize=None)
def _static_rotate_kernel(distance: int):
    @cuda.jit
    def kernel(source, observed):
        thread = cuda.threadIdx.x
        observed[thread] = qualified_coop.shuffle(
            qualified_coop.this_block(),
            source[thread],
            mode="rotate",
            distance=distance,
        )

    return kernel


@cuda.jit
def _runtime_rotate_kernel(source, distances, observed):
    thread = cuda.threadIdx.x
    observed[thread] = qualified_coop.shuffle(
        qualified_coop.this_block(),
        source[thread],
        mode="rotate",
        distance=distances[thread],
    )


@pytest.mark.parametrize("distance_dtype", (np.uint32, np.int64))
def test_scalar_rotate_supports_static_and_per_thread_runtime_distances(
    distance_dtype,
) -> None:
    source = _values(_BLOCK_THREADS, shift=191)
    static_distance = 7
    runtime_distances = ((np.arange(_BLOCK_THREADS) % 11) + 1).astype(distance_dtype)
    static_observed = np.full(_BLOCK_THREADS, -2071, dtype=np.int32)
    runtime_observed = np.full(_BLOCK_THREADS, -2073, dtype=np.int32)
    static_expected = np.empty_like(source)
    runtime_expected = np.empty_like(source)
    for thread in range(_BLOCK_THREADS):
        static_expected[thread] = source[(thread + static_distance) % _BLOCK_THREADS]
        runtime_expected[thread] = source[
            (thread + int(runtime_distances[thread])) % _BLOCK_THREADS
        ]

    _static_rotate_kernel(static_distance)[1, _BLOCK_THREADS](
        source,
        static_observed,
    )
    _runtime_rotate_kernel[1, _BLOCK_THREADS](
        source,
        runtime_distances,
        runtime_observed,
    )

    np.testing.assert_array_equal(static_observed, static_expected)
    np.testing.assert_array_equal(runtime_observed, runtime_expected)


@cuda.jit
def _repeated_storage_reuse_kernel(source, exchange_observed, shuffle_observed):
    thread = cuda.threadIdx.x
    payload = qualified_coop.ThreadData(
        _ITEMS_PER_THREAD,
        dtype=types.int32,
    )
    for item in range(_ITEMS_PER_THREAD):
        payload[item] = source[thread * _ITEMS_PER_THREAD + item]
    striped = qualified_coop.exchange(
        qualified_coop.this_block(),
        payload,
        mode="blocked_to_striped",
    )
    blocked = qualified_coop.exchange(
        qualified_coop.this_block(),
        striped,
        mode="striped_to_blocked",
    )
    for item in range(_ITEMS_PER_THREAD):
        exchange_observed[thread * _ITEMS_PER_THREAD + item] = blocked[item]

    first = qualified_coop.shuffle(
        qualified_coop.this_block(),
        source[thread],
        mode="rotate",
        distance=5,
    )
    shuffle_observed[thread] = qualified_coop.shuffle(
        qualified_coop.this_block(),
        first,
        mode="rotate",
        distance=9,
    )


def test_repeated_exchange_and_shuffle_calls_reuse_implementation_storage() -> None:
    source = _values(_TILE_ITEMS, shift=211)
    exchange_observed = np.full(_TILE_ITEMS, -2081, dtype=np.int32)
    shuffle_observed = np.full(_BLOCK_THREADS, -2083, dtype=np.int32)

    _repeated_storage_reuse_kernel[1, _BLOCK_THREADS](
        source,
        exchange_observed,
        shuffle_observed,
    )

    np.testing.assert_array_equal(exchange_observed, source)
    scalar_source = source[:_BLOCK_THREADS]
    np.testing.assert_array_equal(
        shuffle_observed,
        np.roll(scalar_source, -14),
    )


def _run_invalid_runtime_shuffle_probe(
    mode: str,
    distance: int,
    dtype: str,
) -> subprocess.CompletedProcess[str]:
    # A device trap poisons its CUDA context, so invalid launches must run in
    # isolated processes. The safe-path flag also proves the installed package
    # is used instead of a source-tree package found through the current path.
    script = f"""\
import numpy as np
import numba_cuda_mlir.cuda as cuda
from pathlib import Path

import cuda.coop.numba_mlir as qualified_coop

expected_origin = Path({str(_QUALIFIED_COOP_ORIGIN)!r})
actual_origin = Path(qualified_coop.__file__).resolve()
if actual_origin != expected_origin:
    raise RuntimeError(
        f"trap probe imported cuda.coop from {{actual_origin}}, "
        f"expected {{expected_origin}}"
    )

BLOCK_THREADS = {_BLOCK_THREADS}

@cuda.jit
def kernel(source, distances, observed):
    thread = cuda.threadIdx.x
    observed[thread] = qualified_coop.shuffle(
        qualified_coop.this_block(),
        source[thread],
        mode={mode!r},
        distance=distances[thread],
    )

source = np.arange(BLOCK_THREADS, dtype=np.int32)
distances = np.full(BLOCK_THREADS, {distance}, dtype=np.{dtype})
observed = np.full(BLOCK_THREADS, -1, dtype=np.int32)
kernel[1, BLOCK_THREADS](source, distances, observed)
cuda.synchronize()
raise AssertionError("invalid runtime Shuffle distance did not trap")
"""
    return subprocess.run(
        [sys.executable, _SAFE_PATH_FLAG, "-B", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.mark.parametrize(
    ("distance", "dtype"),
    (
        pytest.param(0, "int32", id="zero"),
        pytest.param(_BLOCK_THREADS, "int32", id="block-threads"),
        pytest.param(1 << 40, "int64", id="wide-int64"),
    ),
)
def test_invalid_runtime_rotate_distance_traps_in_an_isolated_context(
    distance: int,
    dtype: str,
) -> None:
    result = _run_invalid_runtime_shuffle_probe("rotate", distance, dtype)
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output

    source = _values(_BLOCK_THREADS, shift=229)
    observed = np.full(_BLOCK_THREADS, -2091, dtype=np.int32)
    _static_rotate_kernel(3)[1, _BLOCK_THREADS](source, observed)
    np.testing.assert_array_equal(observed, np.roll(source, -3))


@pytest.mark.parametrize(
    "distance",
    (
        pytest.param(-(1 << 40), id="below-int32"),
        pytest.param(1 << 40, id="above-int32"),
    ),
)
def test_runtime_offset_outside_signed_int32_traps_in_an_isolated_context(
    distance: int,
) -> None:
    result = _run_invalid_runtime_shuffle_probe("offset", distance, "int64")
    output = result.stdout + result.stderr

    assert result.returncode != 0, output
    assert any(
        error in output
        for error in (
            "CUDA_ERROR_ILLEGAL_INSTRUCTION",
            "CUDA_ERROR_LAUNCH_FAILED",
        )
    ), output
