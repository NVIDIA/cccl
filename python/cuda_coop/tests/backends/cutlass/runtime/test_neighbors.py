# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent NumPy oracles for block differences and boundary flags."""

import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
    values_for,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


class _Readonly:
    def __init__(self, source):
        self.items_per_thread, self.dtype, self.alignment = (
            source.items_per_thread,
            source.dtype,
            source.alignment,
        )
        self._items = tuple(source)

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


def _run(
    api=coop,
    *,
    dtype=np.int32,
    operation="adjacent_difference",
    mode="left",
    count=None,
    boundary=False,
    payload="thread_data",
    inferred=False,
    block=(8, 4, 2),
    reuse=False,
    sharing="shared",
    manual_sync=False,
    alignment=64,
    capacity=None,
    compile_options=(),
):
    value_type = cutlass_dtype(dtype)
    result_type = value_type if operation == "adjacent_difference" else cutlass.Int32
    items, blocks = 3, 2
    tile = int(np.prod(block)) * items
    size = blocks * tile
    predecessor = (
        7 if boundary and mode in {"left", "heads", "heads_and_tails"} else None
    )
    successor = (
        11 if boundary and mode in {"right", "tails", "heads_and_tails"} else None
    )

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        output: cute.Pointer,
        second: cute.Pointer,
        original: cute.Pointer,
        valid: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        offset = cute.arch.block_idx()[0] * tile + thread * items
        sources = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout(size)), result_type
        )
        seconds = cute.recast_tensor(
            cute.make_tensor(second, cute.make_layout(size)), result_type
        )
        originals = cute.recast_tensor(
            cute.make_tensor(original, cute.make_layout(size)), value_type
        )
        group = api.this_block()
        values = api.ThreadData(
            items, dtype=None if inferred else value_type, alignment=alignment
        )
        for item in cutlass.range_constexpr(items):
            values[item] = sources[offset + item]
        if cutlass.const_expr(payload == "readonly"):
            inputs = _Readonly(values)
        elif cutlass.const_expr(payload == "tensor"):
            inputs = values.to_register_tensor()
        else:
            inputs = values
        if cutlass.const_expr(reuse):
            storage = api.TempStorage(
                size_in_bytes=capacity,
                alignment=alignment,
                sharing=sharing,
                auto_sync=not manual_sync,
            )
        else:
            storage = None
        first = api.ThreadData(items, dtype=result_type)
        other = api.ThreadData(items, dtype=result_type)
        for item in cutlass.range_constexpr(items):
            first[item] = result_type(0)
            other[item] = result_type(0)
        for iteration in range(repeats):
            if cutlass.const_expr(operation == "adjacent_difference"):
                first = api.adjacent_difference(
                    group,
                    inputs,
                    direction=mode,
                    valid_items=valid if count is not None else None,
                    tile_predecessor_item=predecessor,
                    tile_successor_item=successor,
                    temp_storage=storage,
                )
            elif cutlass.const_expr(mode == "heads_and_tails"):
                first, other = api.discontinuity(
                    group,
                    inputs,
                    mode=mode,
                    tile_predecessor_item=predecessor,
                    tile_successor_item=successor,
                    temp_storage=storage,
                )
            else:
                first = api.discontinuity(
                    group,
                    inputs,
                    mode=mode,
                    tile_predecessor_item=predecessor,
                    tile_successor_item=successor,
                    temp_storage=storage,
                )
            if cutlass.const_expr(reuse and manual_sync):
                storage.sync()
        for item in cutlass.range_constexpr(items):
            outputs[offset + item] = first[item]
            seconds[offset + item] = other[item]
            originals[offset + item] = values[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        output: cute.Pointer,
        second: cute.Pointer,
        original: cute.Pointer,
        valid: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        kernel(source, output, second, original, valid, repeats).launch(
            grid=blocks, block=block
        )

    source = np.repeat(values_for(dtype, (size + 3) // 4, shift=17), 4)[:size].copy()
    output_dtype = dtype if operation == "adjacent_difference" else np.int32
    observed = np.zeros(size, dtype=output_dtype)
    secondary = np.zeros_like(observed)
    original = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(secondary) as tail,
        device_array(original) as keep,
    ):
        args = (
            src,
            out,
            tail,
            keep,
            cutlass.Int64(tile if count is None else count),
            cutlass.Int32(5 if reuse else 1),
        )
        compiled = (
            cute.compile[compile_options](launch, *args)
            if compile_options
            else cute.compile(launch, *args)
        )
        compiled(*args)
    np.testing.assert_array_equal(original, source)
    for start in range(0, size, tile):
        values = source[start : start + tile]
        if operation == "adjacent_difference":
            expected = values.copy()
            limit = tile if count is None else count
            with np.errstate(over="ignore"):
                if mode == "left":
                    expected[1:limit] = values[1:limit] - values[: max(0, limit - 1)]
                    if predecessor is not None and limit:
                        expected[0] = values[0] - np.dtype(dtype).type(predecessor)
                else:
                    expected[: max(0, limit - 1)] = (
                        values[: max(0, limit - 1)] - values[1:limit]
                    )
                    if successor is not None and limit:
                        expected[limit - 1] = values[limit - 1] - np.dtype(dtype).type(
                            successor
                        )
            np.testing.assert_array_equal(observed[start : start + tile], expected)
        else:
            heads, tails = np.ones(tile, dtype=np.int32), np.ones(tile, dtype=np.int32)
            heads[1:] = values[1:] != values[:-1]
            tails[:-1] = values[:-1] != values[1:]
            if predecessor is not None:
                heads[0] = values[0] != predecessor
            if successor is not None:
                tails[-1] = values[-1] != successor
            np.testing.assert_array_equal(
                observed[start : start + tile], tails if mode == "tails" else heads
            )
            if mode == "heads_and_tails":
                np.testing.assert_array_equal(secondary[start : start + tile], tails)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize(
    "operation,mode",
    [
        ("adjacent_difference", "left"),
        ("adjacent_difference", "right"),
        ("discontinuity", "heads"),
        ("discontinuity", "tails"),
        ("discontinuity", "heads_and_tails"),
    ],
)
def test_numeric_results(api, dtype, operation, mode):
    _run(api, dtype=dtype, operation=operation, mode=mode)


@pytest.mark.parametrize("mode", ("left", "right"))
@pytest.mark.parametrize("count", (0, 1, 47, 191, 192))
def test_valid_prefix(mode, count):
    _run(mode=mode, count=count)


@pytest.mark.parametrize(
    "operation,mode",
    [
        ("adjacent_difference", "left"),
        ("adjacent_difference", "right"),
        ("discontinuity", "heads"),
        ("discontinuity", "tails"),
        ("discontinuity", "heads_and_tails"),
    ],
)
def test_boundary(operation, mode):
    _run(operation=operation, mode=mode, boundary=True)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize(
    "operation,mode",
    [("adjacent_difference", "left"), ("discontinuity", "heads_and_tails")],
)
def test_readonly_inference(api, operation, mode):
    _run(api, operation=operation, mode=mode, payload="readonly", inferred=True)


def test_qualified_tensor():
    _run(
        cutlass_coop,
        payload="tensor",
        operation="discontinuity",
        mode="heads_and_tails",
    )


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
@pytest.mark.parametrize(
    "operation,mode",
    [("adjacent_difference", "left"), ("discontinuity", "heads_and_tails")],
)
def test_scratch_reuse(sharing, manual_sync, operation, mode):
    _run(
        sharing=sharing,
        manual_sync=manual_sync,
        operation=operation,
        mode=mode,
        reuse=True,
        alignment=128,
    )


def test_alignment_minimum():
    _run(reuse=True, alignment=1, capacity=8192)


def test_undersized_storage():
    with pytest.raises(
        (ValueError, DSLRuntimeError), match="size|capacity|bytes|storage"
    ):
        _run(reuse=True, capacity=1)


def test_final_cubin(tmp_path):
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))))
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        assert "cuda_coop_cutlass_adjacent_difference_" not in sass
        assert re.search(r"\bCALL\b", sass) is None


@pytest.mark.parametrize("count", (-1, 193, 1 << 32))
def test_runtime_count_traps(count):
    script = (
        "from tests.backends.cutlass.runtime.test_neighbors import _run\n"
        + f"_run(count={count})\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert any(
        error in output
        for error in (
            "ILLEGAL_INSTRUCTION",
            "LAUNCH_FAILED",
            "illegal instruction",
            "launch failed",
            "CUDA Driver call failed: 715",
            "CUDA Driver call failed: 719",
        )
    ), output


def test_documented_neighbor_composition():
    # docs: start cutlass-neighbors
    @cute.kernel
    def compare_neighbors(
        source: cute.Pointer,
        deltas: cute.Pointer,
        head_flags: cute.Pointer,
        tail_flags: cute.Pointer,
    ):
        block = coop.this_block()
        values = coop.ThreadData(2, dtype=cutlass.Int32)
        coop.load(block, source, values)
        scratch = coop.TempStorage(alignment=16)
        differences = coop.adjacent_difference(block, values, temp_storage=scratch)
        heads, tails = coop.discontinuity(
            block, values, mode="heads_and_tails", temp_storage=scratch
        )
        coop.store(block, deltas, differences)
        coop.store(block, head_flags, heads)
        coop.store(block, tail_flags, tails)

    @cute.jit
    def launch(
        source: cute.Pointer,
        deltas: cute.Pointer,
        head_flags: cute.Pointer,
        tail_flags: cute.Pointer,
    ):
        compare_neighbors(source, deltas, head_flags, tail_flags).launch(
            grid=1, block=128
        )

    # docs: end cutlass-neighbors

    source = (np.arange(256, dtype=np.int32) // 3) * 7
    deltas, heads, tails = (np.zeros_like(source) for _ in range(3))
    with (
        device_array(source) as src,
        device_array(deltas) as out,
        device_array(heads) as head,
        device_array(tails) as tail,
    ):
        launch(src, out, head, tail)
    expected = np.empty_like(source)
    expected[0], expected[1:] = source[0], np.diff(source)
    np.testing.assert_array_equal(deltas, expected)
    np.testing.assert_array_equal(heads, np.r_[1, source[1:] != source[:-1]])
    np.testing.assert_array_equal(tails, np.r_[source[:-1] != source[1:], 1])
