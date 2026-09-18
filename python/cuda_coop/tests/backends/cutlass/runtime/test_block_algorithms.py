# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Block movement layouts and scratch reuse, checked against scalar oracles."""

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

_ALGORITHMS = (
    "direct",
    "striped",
    "vectorize",
    "transpose",
    "warp_transpose",
    "warp_transpose_timesliced",
)
_SCRATCH_ALGORITHMS = _ALGORITHMS[3:]
_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 4
_TILE = _THREADS * _ITEMS


def _tile_index(algorithm, thread, item):
    return (
        thread + item * _THREADS if algorithm == "striped" else thread * _ITEMS + item
    )


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid", (0, _TILE - 19, _TILE), ids=("zero", "partial", "full")
)
def test_load_layout_and_runtime_bounds(api, algorithm, valid):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        payload = api.ThreadData(_ITEMS)
        api.load(
            api.this_block(),
            source,
            payload,
            algorithm=algorithm,
            valid_items=count,
            oob_default=-29,
            offset=3,
        )
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int32):
        kernel(source, observed, count).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE + 3, shift=17)
    observed = np.full(_TILE, 71, dtype=np.int32)
    expected = np.full(_TILE, -29, dtype=np.int32)
    for thread in range(_THREADS):
        for item in range(_ITEMS):
            index = _tile_index(algorithm, thread, item)
            if index < valid:
                expected[thread * _ITEMS + item] = source[3 + index]
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, valid)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "valid", (0, _TILE - 19, _TILE), ids=("zero", "partial", "full")
)
def test_store_layout_bounds_and_payload_preservation(api, algorithm, valid):
    @cute.kernel
    def kernel(
        source: cute.Pointer, destination: cute.Pointer, preserved: cute.Pointer
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        api.store(
            api.this_block(),
            destination,
            payload,
            algorithm=algorithm,
            valid_items=valid,
            offset=5,
        )
        outputs = cute.make_tensor(preserved, cute.make_layout(_TILE))
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer, destination: cute.Pointer, preserved: cute.Pointer
    ):
        kernel(source, destination, preserved).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE, shift=37)
    destination = np.full(_TILE + 11, -41, dtype=np.int32)
    preserved = np.full(_TILE, 73, dtype=np.int32)
    expected = destination.copy()
    for thread in range(_THREADS):
        for item in range(_ITEMS):
            index = _tile_index(algorithm, thread, item)
            if index < valid:
                expected[5 + index] = source[thread * _ITEMS + item]
    with (
        device_array(source) as src,
        device_array(destination) as dst,
        device_array(preserved) as check,
    ):
        launch(src, dst, check)
    np.testing.assert_array_equal(destination, expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("algorithm", _SCRATCH_ALGORITHMS)
@pytest.mark.parametrize("static_count", (False, True), ids=("runtime", "static"))
def test_partial_transpose_preserves_each_invalid_register(algorithm, static_count):
    valid = _TILE - 19

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(initial, cute.make_layout(_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        payload = cutlass_coop.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        if cutlass.const_expr(static_count):
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=valid,
            )
        else:
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                valid_items=count,
            )
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
    ):
        kernel(source, initial, observed, count).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE)
    initial = -1000 - np.arange(_TILE, dtype=np.int32)
    observed = np.full(_TILE, 71, dtype=np.int32)
    expected = initial.copy()
    expected[:valid] = source[:valid]
    with (
        device_array(source) as src,
        device_array(initial) as seed,
        device_array(observed) as out,
    ):
        launch(src, seed, out, valid)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("offset", (0, 1), ids=("aligned", "misaligned-fallback"))
@pytest.mark.parametrize("algorithm", ("vectorize", "warp_transpose_timesliced"))
def test_unguarded_full_tiles_use_the_correct_layout(algorithm, offset):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_TILE + 1))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        loaded = cutlass_coop.ThreadData(_ITEMS)
        cutlass_coop.load(
            cutlass_coop.this_block(),
            source,
            loaded,
            algorithm=algorithm,
            offset=offset,
        )
        stored = cutlass_coop.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = loaded[item]
            stored[item] = inputs[thread * _ITEMS + item]
        cutlass_coop.store(
            cutlass_coop.this_block(),
            destination,
            stored,
            algorithm=algorithm,
            offset=offset,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, observed: cute.Pointer):
        kernel(source, destination, observed).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE + 1, shift=59)
    destination = np.full(_TILE + 1, -101, dtype=np.int32)
    observed = np.zeros(_TILE, dtype=np.int32)
    expected = destination.copy()
    expected[offset : offset + _TILE] = source[:_TILE]
    with (
        device_array(source) as src,
        device_array(destination) as dst,
        device_array(observed) as out,
    ):
        launch(src, dst, out)
    np.testing.assert_array_equal(observed, source[offset : offset + _TILE])
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("algorithm", _SCRATCH_ALGORITHMS)
@pytest.mark.parametrize("capacity", (None, 16384), ids=("deferred", "fixed"))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True), ids=("auto", "manual"))
def test_storage_reuse_in_runtime_loop(algorithm, capacity, sharing, manual_sync):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        storage = cutlass_coop.TempStorage(
            capacity,
            sharing=sharing,
            auto_sync=False if manual_sync else None,
            alignment=1,
        )
        for tile in range(tiles):
            payload = cutlass_coop.ThreadData(_ITEMS)
            cutlass_coop.load(
                cutlass_coop.this_block(),
                source,
                payload,
                algorithm=algorithm,
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = payload[item] + cutlass.Int32(tile + 1)
            cutlass_coop.store(
                cutlass_coop.this_block(),
                destination,
                payload,
                algorithm=algorithm,
                offset=tile * _TILE,
                temp_storage=storage,
            )
            if cutlass.const_expr(manual_sync):
                storage.sync()

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        kernel(source, destination, tiles).launch(grid=1, block=_BLOCK)

    tiles = 8
    source = values_for(np.int32, tiles * _TILE, shift=61)
    destination = np.full_like(source, -101)
    expected = (
        source.reshape(tiles, _TILE) + np.arange(1, tiles + 1, dtype=np.int32)[:, None]
    )
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, tiles)
    np.testing.assert_array_equal(destination, expected.reshape(-1))


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("alignment", (1, 32, 64))
@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_requested_storage_alignment_is_a_minimum(api, alignment, sharing):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        storage = api.TempStorage(alignment=alignment, sharing=sharing)
        payload = api.ThreadData(_ITEMS)
        api.load(
            api.this_block(),
            source,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )
        api.store(
            api.this_block(),
            destination,
            payload,
            algorithm="transpose",
            temp_storage=storage,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.float64, _TILE, shift=43)
    destination = np.full_like(source, -101)
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
def test_storage_example(sharing, manual_sync):
    path = PACKAGE_ROOT / "examples/cutlass/block_storage.py"
    spec = importlib.util.spec_from_file_location("cutlass_block_storage_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(sharing=sharing, manual_sync=manual_sync)


@pytest.mark.parametrize("algorithm", ("striped", "vectorize", "transpose"))
def test_final_cubin_storage_contract(tmp_path, algorithm):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        storage = cutlass_coop.TempStorage(sharing="shared", alignment=64)
        payload = cutlass_coop.ThreadData(_ITEMS)
        cutlass_coop.load(
            cutlass_coop.this_block(),
            source,
            payload,
            algorithm=algorithm,
            temp_storage=storage,
        )
        cutlass_coop.store(
            cutlass_coop.this_block(),
            destination,
            payload,
            algorithm=algorithm,
            temp_storage=storage,
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE, shift=67)
    destination = np.full_like(source, -101)
    with device_array(source) as src, device_array(destination) as dst:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, dst)
        compiled(src, dst)
    np.testing.assert_array_equal(destination, source)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins, "CUTLASS did not retain the final linked cubin"
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
        # Generic LD/ST instructions can address the shared-memory window;
        # the final cubin's allocation report is authoritative for capacity.
        shared_sizes = [int(size) for size in re.findall(r"\bSHARED:(\d+)", resources)]
        assert shared_sizes
        has_shared = any(shared_sizes)
        has_barrier = re.search(r"\bBAR(?:\.[A-Z0-9_]+)*\b", sass) is not None
        assert has_shared is (algorithm == "transpose")
        assert has_barrier is (algorithm == "transpose")
