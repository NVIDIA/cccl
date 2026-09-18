# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Logical-Warp layouts, independent group controls, and scratch reuse."""

import importlib.util
import re
import shutil
import subprocess

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
_WIDTHS = (1, 2, 4, 8, 16, 32)
_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 4
_BLOCK_TILE = _THREADS * _ITEMS


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("algorithm", ("direct", "transpose"))
@pytest.mark.parametrize("operation", ("load", "store"))
def test_width_layout(api, width, algorithm, operation):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_BLOCK_TILE + 3))
        outputs = cute.make_tensor(destination, cute.make_layout(_BLOCK_TILE + 3))
        group = api.this_warp().group_by(width)
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        if cutlass.const_expr(operation == "load"):
            returned = api.load(group, inputs, payload, algorithm=algorithm, offset=3)
            assert returned is None
            for item in cutlass.range_constexpr(_ITEMS):
                outputs[thread * _ITEMS + item] = payload[item]
        else:
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = inputs[thread * _ITEMS + item]
            api.store(group, outputs, payload, algorithm=algorithm, offset=3)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE + 3, shift=31)
    destination = np.full_like(source, -101)
    expected = destination.copy()
    if operation == "load":
        expected[:_BLOCK_TILE] = source[3:]
    else:
        expected[3:] = source[:_BLOCK_TILE]
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("exhaustive", (False, True))
def test_group_controls(api, algorithm, exhaustive):
    width = 8
    groups = _THREADS // width
    tile = width * _ITEMS
    allocation = _BLOCK_TILE + groups * 3

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        counts: cute.Pointer,
        offsets: cute.Pointer,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        index = thread // width
        valid = cute.make_tensor(counts, cute.make_layout(groups))[index]
        offset = cute.make_tensor(offsets, cute.make_layout(groups))[index]
        seeds = cute.make_tensor(initial, cute.make_layout(_BLOCK_TILE))
        results = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        checks = cute.make_tensor(preserved, cute.make_layout(_BLOCK_TILE))
        group = api.this_warp().group_by(width, exhaustive=exhaustive)
        loaded = api.ThreadData(_ITEMS)
        api.load(
            group,
            source,
            loaded,
            algorithm=algorithm,
            valid_items=valid,
            oob_default=cutlass.Int32(-200 - index),
            offset=offset,
        )
        stored = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            results[thread * _ITEMS + item] = loaded[item]
            stored[item] = seeds[thread * _ITEMS + item]
        api.store(
            group,
            destination,
            stored,
            algorithm=algorithm,
            valid_items=valid,
            offset=offset,
        )
        for item in cutlass.range_constexpr(_ITEMS):
            checks[thread * _ITEMS + item] = stored[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        counts: cute.Pointer,
        offsets: cute.Pointer,
    ):
        kernel(
            source, initial, observed, destination, preserved, counts, offsets
        ).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, allocation, shift=41)
    initial = values_for(np.int32, _BLOCK_TILE, shift=59)
    observed = np.zeros(_BLOCK_TILE, dtype=np.int32)
    destination = np.full(allocation, -101, dtype=np.int32)
    preserved = np.zeros_like(initial)
    counts = np.array([0, 1, 7, tile, 19, tile - 1, 3, 17], dtype=np.int32)
    offsets = 2 + np.arange(groups, dtype=np.int64) * 3
    expected_load = np.empty_like(observed)
    expected_store = destination.copy()
    for thread in range(_THREADS):
        group, lane = divmod(thread, width)
        for item in range(_ITEMS):
            index = (
                lane + item * width if algorithm == "striped" else lane * _ITEMS + item
            )
            payload_index = thread * _ITEMS + item
            memory_index = offsets[group] + group * tile + index
            if index < counts[group]:
                expected_load[payload_index] = source[memory_index]
                expected_store[memory_index] = initial[payload_index]
            else:
                expected_load[payload_index] = -200 - group
    with (
        device_array(source) as src,
        device_array(initial) as seed,
        device_array(observed) as out,
        device_array(destination) as dst,
        device_array(preserved) as check,
        device_array(counts) as count,
        device_array(offsets) as offset,
    ):
        launch(src, seed, out, dst, check, count, offset)
    np.testing.assert_array_equal(observed, expected_load)
    np.testing.assert_array_equal(destination, expected_store)
    np.testing.assert_array_equal(preserved, initial)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("width", _WIDTHS)
def test_invalid_slots(api, width):
    groups = _THREADS // width
    tile = width * _ITEMS

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        counts: cute.Pointer,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        index = thread // width
        valid = cute.make_tensor(counts, cute.make_layout(groups))[index]
        seeds = cute.make_tensor(initial, cute.make_layout(_BLOCK_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = seeds[thread * _ITEMS + item]
        api.load(
            api.this_warp().group_by(width),
            source,
            payload,
            algorithm="transpose",
            valid_items=valid,
        )
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        counts: cute.Pointer,
    ):
        kernel(source, initial, observed, counts).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE, shift=73)
    initial = -1000 - np.arange(_BLOCK_TILE, dtype=np.int32)
    observed = np.zeros_like(initial)
    counts = (np.arange(groups, dtype=np.int32) * 7 + 3) % (tile + 1)
    expected = initial.copy()
    for group in range(groups):
        start = group * tile
        expected[start : start + counts[group]] = source[start : start + counts[group]]
    with (
        device_array(source) as src,
        device_array(initial) as seed,
        device_array(observed) as out,
        device_array(counts) as count,
    ):
        launch(src, seed, out, count)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 8, 32))
@pytest.mark.parametrize(
    "divergent", (False, True), ids=("all-groups", "selected-group")
)
def test_transpose_loop(api, width, divergent):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        group_index = thread // width
        if cutlass.const_expr(width == 32):
            selected = group_index == 1
        else:
            selected = (thread % 32) // width == 2
        if selected or not divergent:
            group = api.this_warp().group_by(width)
            for tile in range(tiles):
                payload = api.ThreadData(_ITEMS)
                api.load(
                    group,
                    source,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )
                for item in cutlass.range_constexpr(_ITEMS):
                    payload[item] = payload[item] + cutlass.Int32(
                        tile + group_index + 1
                    )
                api.store(
                    group,
                    destination,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        kernel(source, destination, tiles).launch(grid=1, block=_BLOCK)

    tiles = 8
    source = values_for(np.int32, tiles * _BLOCK_TILE, shift=83)
    destination = np.full_like(source, -101)
    expected = destination.copy()
    group_tile = width * _ITEMS
    for tile in range(tiles):
        for group in range(_THREADS // width):
            selected = group == 1 if width == 32 else group % (32 // width) == 2
            if divergent and not selected:
                continue
            start = tile * _BLOCK_TILE + group * group_tile
            expected[start : start + group_tile] = (
                source[start : start + group_tile] + tile + group + 1
            )
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, tiles)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize(
    "width,algorithm",
    [(8, algorithm) for algorithm in _ALGORITHMS]
    + [(1, "transpose"), (32, "transpose")],
)
def test_final_cubin(tmp_path, width, algorithm):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        group = cutlass_coop.this_warp().group_by(width)
        payload = cutlass_coop.ThreadData(_ITEMS)
        cutlass_coop.load(group, source, payload, algorithm=algorithm)
        cutlass_coop.store(group, destination, payload, algorithm=algorithm)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE, shift=89)
    destination = np.zeros_like(source)
    with device_array(source) as src, device_array(destination) as dst:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, dst)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        resources = subprocess.check_output(
            [cuobjdump, "--dump-resource-usage", str(cubin)], text=True
        )
        cubin.with_suffix(".sass").write_text(sass)
        cubin.with_suffix(".resources").write_text(resources)
        assert "cuda_coop_cutlass_load_" not in sass
        assert "cuda_coop_cutlass_store_" not in sass
        assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
        assert re.search(r"\bBAR(?:\.[A-Z0-9_]+)*\b", sass) is None
        shared = [int(size) for size in re.findall(r"\bSHARED:(\d+)", resources)]
        assert shared
        if algorithm != "transpose":
            assert not any(shared)
            assert "WARPSYNC" not in sass
        elif width > 1:
            assert any(shared)


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/logical_warp_load_store.py"
    spec = importlib.util.spec_from_file_location("cutlass_logical_warp_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)
