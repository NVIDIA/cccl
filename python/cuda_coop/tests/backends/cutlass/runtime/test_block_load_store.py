# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent numerical oracles for direct CUTLASS Block Load and Store."""

import importlib.util
import math
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
_BLOCK = (8, 4, 1)
_THREADS = 32
_ITEMS = 2
_TILE = _THREADS * _ITEMS


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("operation", ("load", "store"))
def test_direct_layout_matches_independent_oracle(api, dtype, operation):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        tx, ty, tz = cute.arch.thread_idx()
        thread = tx + _BLOCK[0] * (ty + _BLOCK[1] * tz)
        # CuTe's signless tensor construction needs an explicit unsigned view.
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_TILE)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(destination, cute.make_layout(_TILE)), value_type
        )
        payload = api.ThreadData(_ITEMS, dtype=value_type)
        if cutlass.const_expr(operation == "load"):
            returned = api.load(api.this_block(), inputs, payload)
            assert returned is None
            for item in cutlass.range_constexpr(_ITEMS):
                outputs[thread * _ITEMS + item] = payload[item]
        else:
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = inputs[thread * _ITEMS + item]
            api.store(api.this_block(), outputs, payload)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(dtype, _TILE, shift=13)
    destination = np.zeros_like(source)
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst)
    np.testing.assert_array_equal(destination, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("valid", (0, 35, _TILE))
@pytest.mark.parametrize("runtime_valid", (False, True), ids=("static", "runtime"))
@pytest.mark.parametrize("default", (None, -7), ids=("preserve", "default"))
def test_partial_load_preserves_or_fills_invalid_items(
    api, valid, runtime_valid, default
):
    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int32):
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = cutlass.Int32(113 + item)
        if cutlass.const_expr(runtime_valid):
            api.load(
                api.this_block(),
                source,
                payload,
                valid_items=count,
                oob_default=default,
                offset=3,
            )
        else:
            api.load(
                api.this_block(),
                source,
                payload,
                valid_items=valid,
                oob_default=default,
                offset=3,
            )
        api.store(api.this_block(), destination, payload)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int32):
        kernel(source, destination, count).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE + 3)
    destination = np.zeros(_TILE, dtype=np.int32)
    expected = np.tile(np.array([113, 114], dtype=np.int32), _THREADS)
    if default is not None:
        expected.fill(default)
    expected[:valid] = source[3 : 3 + valid]
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, valid)
    np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("valid", (0, 35, _TILE))
@pytest.mark.parametrize("runtime_valid", (False, True), ids=("static", "runtime"))
def test_partial_store_respects_runtime_offsets_and_tail(valid, runtime_valid):
    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        count: cutlass.Int32,
        offset: cutlass.Int64,
    ):
        payload = coop.ThreadData(_ITEMS)
        coop.load(coop.this_block(), source, payload)
        if cutlass.const_expr(runtime_valid):
            coop.store(
                coop.this_block(),
                destination,
                payload,
                valid_items=count,
                offset=offset,
            )
        else:
            coop.store(
                coop.this_block(),
                destination,
                payload,
                valid_items=valid,
                offset=offset,
            )

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        count: cutlass.Int32,
        offset: cutlass.Int64,
    ):
        kernel(source, destination, count, offset).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE)
    destination = np.full(_TILE + 11, -101, dtype=np.int32)
    expected = destination.copy()
    expected[5 : 5 + valid] = source[:valid]
    with device_array(source) as src, device_array(destination) as dst:
        launch(src, dst, valid, 5)
    np.testing.assert_array_equal(destination, expected)


def test_runtime_load_default_and_offset_change_between_launches():
    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        count: cutlass.Int32,
        default: cutlass.Int32,
        offset: cutlass.Int64,
    ):
        payload = coop.ThreadData(_ITEMS)
        coop.load(
            coop.this_block(),
            source,
            payload,
            valid_items=count,
            oob_default=default,
            offset=offset,
        )
        coop.store(coop.this_block(), destination, payload)

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        count: cutlass.Int32,
        default: cutlass.Int32,
        offset: cutlass.Int64,
    ):
        kernel(source, destination, count, default, offset).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE + 7)
    for count, default, offset in ((0, -17, 3), (27, -23, 5), (_TILE, -31, 7)):
        destination = np.zeros(_TILE, dtype=np.int32)
        expected = np.full(_TILE, default, dtype=np.int32)
        expected[:count] = source[offset : offset + count]
        with device_array(source) as src, device_array(destination) as dst:
            launch(src, dst, count, default, offset)
        np.testing.assert_array_equal(destination, expected)


@pytest.mark.parametrize("block", ((32, 1, 1), (8, 4, 1), (4, 4, 2)))
@pytest.mark.parametrize("alignment", (1, 4, 16, 32))
def test_scalar_store_and_payload_alignment(block, alignment):
    @cute.kernel
    def kernel(destination: cute.Pointer):
        tx, ty, tz = cute.arch.thread_idx()
        thread = tx + block[0] * (ty + block[1] * tz)
        payload = cutlass_coop.ThreadData(1, dtype=cutlass.Int32, alignment=alignment)
        payload[0] = cutlass.Int32(thread * 3 + 1)
        cutlass_coop.store(cutlass_coop.this_block(), destination, payload)
        cutlass_coop.load(cutlass_coop.this_block(), destination, payload)
        cutlass_coop.store(
            cutlass_coop.this_block(), destination, payload, offset=math.prod(block)
        )
        cutlass_coop.store(
            cutlass_coop.this_block(),
            destination,
            cutlass.Int32(thread + 7),
            offset=2 * math.prod(block),
        )

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=block)

    count = math.prod(block)
    destination = np.zeros(count * 3, dtype=np.int32)
    with device_array(destination) as dst:
        launch(dst)
    np.testing.assert_array_equal(destination[:count], np.arange(count) * 3 + 1)
    np.testing.assert_array_equal(
        destination[count : 2 * count], np.arange(count) * 3 + 1
    )
    np.testing.assert_array_equal(destination[2 * count :], np.arange(count) + 7)


def test_final_cubin_eliminates_direct_providers_and_scratch(tmp_path):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        payload = coop.ThreadData(_ITEMS)
        coop.load(coop.this_block(), source, payload)
        coop.store(coop.this_block(), destination, payload)

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE)
    destination = np.zeros_like(source)
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
        cubin.with_suffix(".sass").write_text(sass)
        assert "cuda_coop_cutlass_load_" not in sass
        assert "cuda_coop_cutlass_store_" not in sass
        assert (
            re.search(r"\b(?:CALL|LDL|STL|LDS|STS|BAR)(?:\.[A-Z0-9_]+)*\b", sass)
            is None
        )
        assert "LDG" in sass and "STG" in sass


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_executable_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/block_load_store.py"
    spec = importlib.util.spec_from_file_location("cutlass_load_store_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)
