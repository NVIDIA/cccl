# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent physical-Warp movement, addressing, and scratch reuse oracles."""

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
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
    values_for,
)
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_ALGORITHMS = ("direct", "striped", "vectorize", "transpose")
_BLOCK = (8, 4, 2)
_THREADS = 64
_WIDTH = 32
_ITEMS = 4
_WARP_TILE = _WIDTH * _ITEMS
_BLOCK_TILE = _THREADS * _ITEMS


def _index(algorithm, lane, item):
    return lane + item * _WIDTH if algorithm == "striped" else lane * _ITEMS + item


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "base_valid", (0, 100, _WARP_TILE), ids=("zero", "partial", "full-first-warp")
)
def test_group_local_load_counts_offsets_and_defaults(api, algorithm, base_valid):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, valid: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        count = valid - warp * 7
        if count < 0:
            count = cutlass.Int32(0)
        payload = api.ThreadData(_ITEMS)
        returned = api.load(
            api.this_warp(),
            source,
            payload,
            algorithm=algorithm,
            valid_items=count,
            oob_default=cutlass.Int32(-29 - warp),
            offset=cutlass.Int64(3 + warp * 2),
        )
        assert returned is None
        outputs = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, valid: cutlass.Int32):
        kernel(source, observed, valid).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE + 7, shift=11)
    observed = np.full(_BLOCK_TILE, 71, dtype=np.int32)
    expected = np.empty_like(observed)
    for thread in range(_THREADS):
        warp, lane = divmod(thread, _WIDTH)
        count = max(base_valid - warp * 7, 0)
        origin = warp * _WARP_TILE + 3 + warp * 2
        for item in range(_ITEMS):
            index = _index(algorithm, lane, item)
            expected[thread * _ITEMS + item] = (
                source[origin + index] if index < count else -29 - warp
            )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, base_valid)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize(
    "base_valid", (0, 100, _WARP_TILE), ids=("zero", "partial", "full-first-warp")
)
def test_group_local_store_counts_offsets_preserve_payload(api, algorithm, base_valid):
    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        valid: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        count = valid - warp * 7
        if count < 0:
            count = cutlass.Int32(0)
        inputs = cute.make_tensor(source, cute.make_layout(_BLOCK_TILE))
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        api.store(
            api.this_warp(),
            destination,
            payload,
            algorithm=algorithm,
            valid_items=count,
            offset=cutlass.Int64(5 + warp * 4),
        )
        outputs = cute.make_tensor(preserved, cute.make_layout(_BLOCK_TILE))
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        preserved: cute.Pointer,
        valid: cutlass.Int32,
    ):
        kernel(source, destination, preserved, valid).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE, shift=19)
    destination = np.full(_BLOCK_TILE + 15, -101, dtype=np.int32)
    preserved = np.zeros_like(source)
    expected = destination.copy()
    for thread in range(_THREADS):
        warp, lane = divmod(thread, _WIDTH)
        count = max(base_valid - warp * 7, 0)
        origin = warp * _WARP_TILE + 5 + warp * 4
        for item in range(_ITEMS):
            index = _index(algorithm, lane, item)
            if index < count:
                expected[origin + index] = source[thread * _ITEMS + item]
    with (
        device_array(source) as src,
        device_array(destination) as dst,
        device_array(preserved) as check,
    ):
        launch(src, dst, check, base_valid)
    np.testing.assert_array_equal(destination, expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("operation", ("load", "store"))
@pytest.mark.parametrize("algorithm", ("direct", "transpose"))
def test_warp_layout_for_every_dtype(api, dtype, operation, algorithm):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_BLOCK_TILE)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(destination, cute.make_layout(_BLOCK_TILE)), value_type
        )
        payload = api.ThreadData(_ITEMS, dtype=value_type)
        if cutlass.const_expr(operation == "load"):
            api.load(api.this_warp(), inputs, payload, algorithm=algorithm)
            for item in cutlass.range_constexpr(_ITEMS):
                outputs[thread * _ITEMS + item] = payload[item]
        else:
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = inputs[thread * _ITEMS + item]
            api.store(api.this_warp(), outputs, payload, algorithm=algorithm)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(dtype, _BLOCK_TILE, shift=29)
    destination = np.zeros_like(source)
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("static_count", (False, True), ids=("runtime", "static"))
def test_partial_transpose_preserves_each_warps_invalid_registers(api, static_count):
    valid = _WARP_TILE - 19

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        initial: cute.Pointer,
        observed: cute.Pointer,
        count: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(initial, cute.make_layout(_BLOCK_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_BLOCK_TILE))
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        if cutlass.const_expr(static_count):
            api.load(
                api.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=valid,
            )
        else:
            api.load(
                api.this_warp(),
                source,
                payload,
                algorithm="transpose",
                valid_items=count - (thread // _WIDTH) * 13,
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

    source = values_for(np.int32, _BLOCK_TILE)
    initial = -1000 - np.arange(_BLOCK_TILE, dtype=np.int32)
    observed = np.zeros_like(initial)
    expected = initial.copy()
    for warp in range(_THREADS // _WIDTH):
        start = warp * _WARP_TILE
        count = valid if static_count else valid - warp * 13
        expected[start : start + count] = source[start : start + count]
    with (
        device_array(source) as src,
        device_array(initial) as seed,
        device_array(observed) as out,
    ):
        launch(src, seed, out, valid)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize(
    "divergent", (False, True), ids=("all-warps", "second-warp-only")
)
def test_transpose_runtime_loop_and_whole_warp_divergence(api, divergent):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        warp = thread // _WIDTH
        if warp == 1 or not divergent:
            for tile in range(tiles):
                payload = api.ThreadData(_ITEMS)
                api.load(
                    api.this_warp(),
                    source,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )
                for item in cutlass.range_constexpr(_ITEMS):
                    payload[item] = payload[item] + cutlass.Int32(tile + warp + 1)
                api.store(
                    api.this_warp(),
                    destination,
                    payload,
                    algorithm="transpose",
                    offset=tile * _BLOCK_TILE,
                )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, tiles: cutlass.Int32):
        kernel(source, destination, tiles).launch(grid=1, block=_BLOCK)

    tiles = 8
    source = values_for(np.int32, tiles * _BLOCK_TILE, shift=43)
    destination = np.full_like(source, -101)
    expected = destination.copy()
    for tile in range(tiles):
        for warp in range(_THREADS // _WIDTH):
            if divergent and warp != 1:
                continue
            start = tile * _BLOCK_TILE + warp * _WARP_TILE
            expected[start : start + _WARP_TILE] = (
                source[start : start + _WARP_TILE] + tile + warp + 1
            )
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, tiles)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("algorithm", _ALGORITHMS)
def test_final_warp_cubin_has_no_block_barrier(tmp_path, algorithm):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        payload = cutlass_coop.ThreadData(_ITEMS)
        cutlass_coop.load(
            cutlass_coop.this_warp(), source, payload, algorithm=algorithm
        )
        cutlass_coop.store(
            cutlass_coop.this_warp(), destination, payload, algorithm=algorithm
        )

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _BLOCK_TILE, shift=59)
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
        assert any(shared) is (algorithm == "transpose")
        if algorithm != "transpose":
            assert "WARPSYNC" not in sass


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_warp_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/warp_load_store.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_warp_load_store_example", path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)
