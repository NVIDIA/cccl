# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Repeated Scan calls and shared scratch across Load, Scan, and Store."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 2
_TILE = _THREADS * _ITEMS
_SETUPS = (
    (None, None, True, 1),
    ("shared", None, True, 1),
    ("shared", None, False, 64),
    ("shared", 8192, True, 64),
    ("shared", 8192, False, 1),
    ("exclusive", None, True, 1),
    ("exclusive", None, False, 64),
    ("exclusive", 8192, True, 64),
    ("exclusive", 8192, False, 1),
)
_SETUP_NAMES = (
    "implicit",
    "shared-inferred",
    "shared-manual",
    "shared-fixed",
    "shared-fixed-manual",
    "exclusive-inferred",
    "exclusive-manual",
    "exclusive-fixed",
    "exclusive-fixed-manual",
)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize(
    "sharing,capacity,auto_sync,alignment", _SETUPS, ids=_SETUP_NAMES
)
def test_mixed_loop(api, sharing, capacity, auto_sync, alignment):
    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        tiles: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        checks = cute.make_tensor(preserved, cute.make_layout(8 * _TILE))
        if cutlass.const_expr(sharing is None):
            storage = None
        else:
            storage = api.TempStorage(
                capacity, sharing=sharing, auto_sync=auto_sync, alignment=alignment
            )
        group = api.this_block()
        for tile in range(tiles):
            payload = api.ThreadData(_ITEMS)
            api.load(
                group,
                source,
                payload,
                algorithm="transpose",
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(not auto_sync):
                storage.sync()
            result = api.exclusive_scan(
                group,
                payload,
                initial_value=7,
                algorithm="raking_memoize",
                temp_storage=storage,
            )
            if cutlass.const_expr(not auto_sync):
                storage.sync()
            api.store(
                group,
                observed,
                result,
                algorithm="transpose",
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(not auto_sync):
                storage.sync()
            for item in cutlass.range_constexpr(_ITEMS):
                checks[tile * _TILE + thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        tiles: cutlass.Int32,
    ):
        kernel(source, observed, preserved, tiles).launch(grid=1, block=_BLOCK)

    tiles = 8
    source = values_for(np.int64, tiles * _TILE, shift=59)
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    expected = np.concatenate(
        [
            np.concatenate(
                (np.array([7], dtype=np.int64), 7 + row[:-1].cumsum(dtype=np.int64))
            )
            for row in source.reshape(tiles, _TILE)
        ]
    )
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check, tiles)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 8, 32))
@pytest.mark.parametrize(
    "divergent", (False, True), ids=("all-groups", "selected-group")
)
def test_warp_loop(api, width, divergent):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        if cutlass.const_expr(width == 32):
            selected = thread // 32 == 1
        else:
            selected = (thread % 32) // width == 2
        if selected or not divergent:
            group = api.this_warp().group_by(width)
            total = cutlass.Int32(0)
            for iteration in range(iterations):
                total += api.inclusive_sum(group, inputs[thread] + iteration)
            outputs[thread] = total

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        kernel(source, observed, iterations).launch(grid=1, block=_BLOCK)

    iterations = 8
    source = values_for(np.int32, _THREADS, shift=67)
    observed = np.full_like(source, -101)
    expected = observed.copy()
    for start in range(0, _THREADS, width):
        selected = start // 32 == 1 if width == 32 else (start % 32) // width == 2
        if divergent and not selected:
            continue
        expected[start : start + width] = iterations * source[
            start : start + width
        ].cumsum(dtype=np.int32) + np.arange(1, width + 1) * sum(range(iterations))
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)
