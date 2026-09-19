# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent Merge Sort ordering, association, and preservation oracles."""

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
        self.items_per_thread = source.items_per_thread
        self.dtype = source.dtype
        self.alignment = source.alignment
        self._items = tuple(source)

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


def _run(
    api=coop,
    *,
    dtype=np.int32,
    value_dtype=np.float64,
    width=64,
    pairs=True,
    partial=False,
    count=None,
    descending=False,
    readonly="none",
    inferred=False,
    block=(8, 4, 2),
    reuse=False,
    sharing="shared",
    manual_sync=False,
    alignment=64,
    capacity=None,
    compile_options=(),
    selected_groups=False,
):
    key_type, value_type = cutlass_dtype(dtype), cutlass_dtype(value_dtype)
    threads = int(np.prod(block))
    items = 3
    size = threads * items
    group_size = threads if width == 64 else width
    tile = group_size * items
    count = tile - 2 if count is None else count
    info = np.finfo(dtype) if np.dtype(dtype).kind == "f" else np.iinfo(dtype)
    sentinel = np.dtype(dtype).type(info.min if descending else info.max).item()

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        check_keys: cute.Pointer,
        check_values: cute.Pointer,
        valid: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        sources = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), key_type
        )
        inputs = cute.recast_tensor(
            cute.make_tensor(payload, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout(size)), key_type
        )
        associated = cute.recast_tensor(
            cute.make_tensor(associations, cute.make_layout(size)), value_type
        )
        key_checks = cute.recast_tensor(
            cute.make_tensor(check_keys, cute.make_layout(size)), key_type
        )
        value_checks = cute.recast_tensor(
            cute.make_tensor(check_values, cute.make_layout(size)), value_type
        )
        if cutlass.const_expr(width == 64):
            group = api.this_block()
        elif cutlass.const_expr(width == 32):
            group = api.this_warp()
        else:
            group = api.this_warp().group_by(width)
        keys = api.ThreadData(
            items, dtype=None if inferred else key_type, alignment=alignment
        )
        values = api.ThreadData(
            items, dtype=None if inferred else value_type, alignment=alignment
        )
        for item in cutlass.range_constexpr(items):
            keys[item] = sources[thread * items + item]
            values[item] = inputs[thread * items + item]
        if cutlass.const_expr(readonly in {"keys", "both"}):
            key_input = _Readonly(keys)
        else:
            key_input = keys
        if cutlass.const_expr(readonly in {"values", "both"}):
            value_input = _Readonly(values)
        else:
            value_input = values
        if cutlass.const_expr(reuse and width == 64):
            storage = api.TempStorage(
                size_in_bytes=capacity,
                alignment=alignment,
                sharing=sharing,
                auto_sync=not manual_sync,
            )
        else:
            storage = None
        result, result_values = keys, values
        for iteration in range(repeats):
            if not selected_groups or (thread // group_size) % 2 == 0:
                if cutlass.const_expr(pairs):
                    if cutlass.const_expr(partial):
                        result, result_values = api.merge_sort_pairs(
                            group,
                            key_input,
                            value_input,
                            descending=descending,
                            valid_items=valid,
                            oob_default=sentinel,
                            temp_storage=storage,
                        )
                    else:
                        result, result_values = api.merge_sort_pairs(
                            group,
                            key_input,
                            value_input,
                            descending=descending,
                            temp_storage=storage,
                        )
                else:
                    if cutlass.const_expr(partial):
                        result = api.merge_sort_keys(
                            group,
                            key_input,
                            descending=descending,
                            valid_items=valid,
                            oob_default=sentinel,
                            temp_storage=storage,
                        )
                    else:
                        result = api.merge_sort_keys(
                            group,
                            key_input,
                            descending=descending,
                            temp_storage=storage,
                        )
                if cutlass.const_expr(reuse and width == 64 and manual_sync):
                    storage.sync()
        for item in cutlass.range_constexpr(items):
            outputs[thread * items + item] = result[item]
            associated[thread * items + item] = result_values[item]
            key_checks[thread * items + item] = keys[item]
            value_checks[thread * items + item] = values[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        payload: cute.Pointer,
        output: cute.Pointer,
        associations: cute.Pointer,
        check_keys: cute.Pointer,
        check_values: cute.Pointer,
        valid: cutlass.Int64,
        repeats: cutlass.Int32,
    ):
        kernel(
            source,
            payload,
            output,
            associations,
            check_keys,
            check_values,
            valid,
            repeats,
        ).launch(grid=1, block=block)

    source = values_for(dtype, size, shift=17)
    payload = (np.arange(size) % 101).astype(value_dtype)
    if np.dtype(value_dtype).kind == "f":
        payload += np.dtype(value_dtype).type(0.25)
    observed, associated = np.zeros_like(source), np.zeros_like(payload)
    preserved_keys, preserved_values = np.zeros_like(source), np.zeros_like(payload)
    with (
        device_array(source) as src,
        device_array(payload) as val,
        device_array(observed) as out,
        device_array(associated) as assoc,
        device_array(preserved_keys) as keys_check,
        device_array(preserved_values) as values_check,
    ):
        args = (
            src,
            val,
            out,
            assoc,
            keys_check,
            values_check,
            cutlass.Int64(count),
            cutlass.Int32(3 if reuse else 1),
        )
        compiled = (
            cute.compile[compile_options](launch, *args)
            if compile_options
            else cute.compile(launch, *args)
        )
        compiled(*args)
    np.testing.assert_array_equal(preserved_keys, source)
    np.testing.assert_array_equal(preserved_values, payload)
    for start in range(0, size, tile):
        if selected_groups and (start // tile) % 2:
            np.testing.assert_array_equal(
                observed[start : start + tile], source[start : start + tile]
            )
            continue
        valid_count = count if partial else tile
        expected = np.sort(source[start : start + valid_count])
        if descending:
            expected = expected[::-1]
        np.testing.assert_array_equal(observed[start : start + valid_count], expected)
        if pairs:
            actual = sorted(
                zip(
                    observed[start : start + valid_count].tolist(),
                    associated[start : start + valid_count].tolist(),
                )
            )
            original = sorted(
                zip(
                    source[start : start + valid_count].tolist(),
                    payload[start : start + valid_count].tolist(),
                )
            )
            assert actual == original


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("pairs", (False, True))
def test_numeric_keys(api, dtype, pairs):
    _run(api, dtype=dtype, pairs=pairs)


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32, 64))
@pytest.mark.parametrize("descending", (False, True))
@pytest.mark.parametrize("prefix", ("zero", "one", "interior", "full"))
def test_group_prefix(width, descending, prefix):
    count = {"zero": 0, "one": 1, "interior": width * 3 - 1, "full": width * 3}[prefix]
    _run(width=width, partial=True, count=count, descending=descending)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
@pytest.mark.parametrize("readonly", ("keys", "values", "both"))
@pytest.mark.parametrize("inferred", (False, True))
def test_readonly_inputs(api, readonly, inferred):
    _run(api, readonly=readonly, inferred=inferred, width=8, partial=True)


@pytest.mark.parametrize("value_dtype", (np.int8, np.uint16, np.float32, np.uint64))
def test_independent_value_type(value_dtype):
    _run(value_dtype=value_dtype, width=32, partial=True)


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("descending", (False, True))
def test_partial_key_types(dtype, descending):
    _run(dtype=dtype, partial=True, descending=descending)


@pytest.mark.parametrize("api", (coop, cutlass_coop), ids=("common", "qualified"))
def test_readonly_keys_only(api):
    _run(api, pairs=False, readonly="keys", inferred=True)


@pytest.mark.parametrize("block", ((16, 1, 1), (8, 8, 1), (4, 4, 4)))
def test_block_dimensions(block):
    _run(block=block, partial=True)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
def test_storage_alignment_minimum(sharing):
    _run(reuse=True, sharing=sharing, alignment=1, capacity=8192)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("manual_sync", (False, True))
def test_reuse_storage(sharing, manual_sync):
    _run(reuse=True, sharing=sharing, manual_sync=manual_sync, alignment=128)


@pytest.mark.parametrize("width", (1, 8, 32))
def test_selected_warp_groups(width):
    _run(width=width, reuse=True, selected_groups=True)


def test_undersized_storage():
    with pytest.raises(
        (ValueError, DSLRuntimeError), match="size|capacity|bytes|storage"
    ):
        _run(reuse=True, capacity=1)


@pytest.mark.parametrize("width", (8, 64))
def test_final_cubin(tmp_path, width):
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(
        width=width,
        partial=True,
        compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))),
    )
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        assert "cuda_coop_cutlass_merge_sort_" not in sass
        assert re.search(r"\bCALL\b", sass) is None
        if width < 64:
            assert re.search(r"\bBAR\.SYNC\b", sass) is None


@pytest.mark.parametrize("count", (-1, 193, 1 << 32))
def test_runtime_count_traps(count):
    script = (
        "from tests.backends.cutlass.runtime.test_merge_sort import _run\n"
        f"_run(partial=True, count={count})\n"
        "raise AssertionError('invalid count did not trap')\n"
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
