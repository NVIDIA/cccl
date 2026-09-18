# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent Exchange layout and scatter oracles with input preservation."""

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

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_BLOCK = (8, 4, 2)
_THREADS = 64
_ITEMS = 3
_TILE = _THREADS * _ITEMS
_LAYOUTS = ("striped_to_blocked", "blocked_to_striped")
_SCATTERS = (
    "scatter_to_blocked",
    "scatter_to_striped",
    "scatter_to_striped_guarded",
    "scatter_to_striped_flagged",
)


def _layout(source, width, mode):
    result = np.empty_like(source)
    for first in range(0, source.size, width * _ITEMS):
        for lane in range(width):
            for item in range(_ITEMS):
                logical = (
                    lane * _ITEMS + item
                    if mode == "striped_to_blocked"
                    else item * width + lane
                )
                source_lane, source_item = (
                    (logical % width, logical // width)
                    if mode == "striped_to_blocked"
                    else divmod(logical, _ITEMS)
                )
                result[first + lane * _ITEMS + item] = source[
                    first + source_lane * _ITEMS + source_item
                ]
    return result


def _run_layout(api, dtype, mode, scope, *, width=8, block=_BLOCK, time_slicing=False):
    value_type = cutlass_dtype(dtype)
    threads = int(np.prod(block))
    size = threads * _ITEMS

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(size)), value_type
        )
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), value_type
        )
        if cutlass.const_expr(scope == "block"):
            group = api.this_block()
        elif cutlass.const_expr(scope == "warp"):
            group = api.this_warp()
        else:
            group = api.this_warp().group_by(width)
        payload = api.ThreadData(_ITEMS, dtype=value_type, alignment=64)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        if cutlass.const_expr(time_slicing):
            result = api.exchange(group, payload, mode=mode, warp_time_slicing=True)
        else:
            result = api.exchange(group, payload, mode=mode)
        assert result.alignment >= 64
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = result[item]
            checks[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        kernel(source, observed, preserved).launch(grid=1, block=block)

    source = values_for(dtype, size, shift=17)
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    group_width = threads if scope == "block" else 32 if scope == "warp" else width
    if "warp_striped" in mode:
        group_width = 32
        oracle_mode = mode.replace("warp_striped", "striped")
    else:
        oracle_mode = mode
    expected = _layout(source, group_width, oracle_mode)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("mode", _LAYOUTS)
@pytest.mark.parametrize("scope", ("block", "warp", "logical"))
def test_layout_types(api, dtype, mode, scope):
    _run_layout(api, dtype, mode, scope)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("mode", _LAYOUTS)
def test_logical_width(api, width, mode):
    _run_layout(api, np.int32, mode, "logical", width=width)


@pytest.mark.parametrize(
    "mode", (*_LAYOUTS, "blocked_to_warp_striped", "warp_striped_to_blocked")
)
@pytest.mark.parametrize("time_slicing", (False, True), ids=("ordinary", "time-sliced"))
def test_block_modes(mode, time_slicing):
    _run_layout(cutlass_coop, np.int32, mode, "block", time_slicing=time_slicing)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("mode", _LAYOUTS)
def test_partial_block(api, mode):
    _run_layout(api, np.int64, mode, "block", block=(8, 3, 2))


def _run_scatter(
    mode,
    *,
    dtype=np.int32,
    rank_dtype=np.int32,
    flag_dtype=np.int8,
    time_slicing=False,
    block=(8, 4, 1),
):
    threads = int(np.prod(block))
    size = threads * _ITEMS
    value_type = cutlass_dtype(dtype)
    rank_type = cutlass_dtype(rank_dtype)
    flag_type = cutlass_dtype(flag_dtype)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        ranks_source: cute.Pointer,
        flags_source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        ranks_out: cute.Pointer,
        flags_out: cute.Pointer,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + block[0] * (y + block[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        rank_inputs = cute.recast_tensor(
            cute.make_tensor(ranks_source, cute.make_layout(size)), rank_type
        )
        flag_inputs = cute.recast_tensor(
            cute.make_tensor(flags_source, cute.make_layout(size)), flag_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(size)), value_type
        )
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), value_type
        )
        rank_checks = cute.recast_tensor(
            cute.make_tensor(ranks_out, cute.make_layout(size)), rank_type
        )
        flag_checks = cute.recast_tensor(
            cute.make_tensor(flags_out, cute.make_layout(size)), flag_type
        )
        payload = cutlass_coop.ThreadData(_ITEMS, dtype=value_type)
        ranks = cutlass_coop.ThreadData(_ITEMS, dtype=rank_type)
        flags = cutlass_coop.ThreadData(_ITEMS, dtype=flag_type)
        for item in cutlass.range_constexpr(_ITEMS):
            index = thread * _ITEMS + item
            payload[item] = inputs[index]
            ranks[item] = rank_inputs[index]
            flags[item] = flag_inputs[index]
        if cutlass.const_expr(mode == "scatter_to_striped_flagged"):
            result = cutlass_coop.exchange(
                cutlass_coop.this_block(),
                payload,
                mode=mode,
                ranks=ranks,
                valid_flags=flags,
            )
        else:
            result = cutlass_coop.exchange(
                cutlass_coop.this_block(),
                payload,
                mode=mode,
                ranks=ranks,
                warp_time_slicing=time_slicing,
            )
        for item in cutlass.range_constexpr(_ITEMS):
            index = thread * _ITEMS + item
            outputs[index] = result[item]
            checks[index] = payload[item]
            rank_checks[index] = ranks[item]
            flag_checks[index] = flags[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        ranks_source: cute.Pointer,
        flags_source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        ranks_out: cute.Pointer,
        flags_out: cute.Pointer,
    ):
        kernel(
            source,
            ranks_source,
            flags_source,
            observed,
            preserved,
            ranks_out,
            flags_out,
        ).launch(grid=1, block=block)

    source = values_for(dtype, size, shift=23)
    ranks = ((np.arange(size) * 17 + 11) % size).astype(rank_dtype)
    flags = np.full(size, 3, dtype=flag_dtype)
    if np.dtype(flag_dtype).kind == "i":
        flags[1::3] = -2
    if mode.endswith("guarded"):
        ranks[::7] = -1
    if mode.endswith("flagged"):
        flags[::7] = 0
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    ranks_out = np.zeros_like(ranks)
    flags_out = np.zeros_like(flags)
    expected = np.empty_like(source)
    defined = np.zeros(size, dtype=bool)
    for index, rank in enumerate(ranks):
        if int(rank) < 0 or (mode.endswith("flagged") and flags[index] == 0):
            continue
        destination = (
            int(rank)
            if mode == "scatter_to_blocked"
            else (int(rank) % threads) * _ITEMS + int(rank) // threads
        )
        expected[destination] = source[index]
        defined[destination] = True
    with (
        device_array(source) as src,
        device_array(ranks) as rank,
        device_array(flags) as flag,
        device_array(observed) as out,
        device_array(preserved) as check,
        device_array(ranks_out) as rank_check,
        device_array(flags_out) as flag_check,
    ):
        launch(src, rank, flag, out, check, rank_check, flag_check)
    np.testing.assert_array_equal(observed[defined], expected[defined])
    np.testing.assert_array_equal(preserved, source)
    np.testing.assert_array_equal(ranks_out, ranks)
    np.testing.assert_array_equal(flags_out, flags)


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("mode", _SCATTERS)
def test_scatter_types(dtype, mode):
    _run_scatter(mode, dtype=dtype)


@pytest.mark.parametrize("rank_dtype", (np.int8, np.int16, np.int32, np.int64))
@pytest.mark.parametrize("mode", _SCATTERS)
def test_rank_types(rank_dtype, mode):
    _run_scatter(mode, rank_dtype=rank_dtype)


@pytest.mark.parametrize("flag_dtype", NUMPY_DTYPES[:8])
def test_flag_types(flag_dtype):
    _run_scatter("scatter_to_striped_flagged", flag_dtype=flag_dtype)


@pytest.mark.parametrize("mode", ("scatter_to_blocked", "scatter_to_striped"))
def test_scatter_slicing(mode):
    _run_scatter(mode, time_slicing=True, block=_BLOCK)


@pytest.mark.parametrize("mode", _SCATTERS)
def test_scatter_partial_block(mode):
    _run_scatter(mode, block=(8, 3, 2))


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("width", (1, 8, 32, 64))
@pytest.mark.parametrize(
    "divergent", (False, True), ids=("all-groups", "selected-group")
)
def test_reuse_loop(api, width, divergent):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        if cutlass.const_expr(width == 64):
            group = api.this_block()
            selected = True
        else:
            group = api.this_warp().group_by(width)
            if cutlass.const_expr(width == 32):
                selected = thread // 32 == 1
            else:
                selected = (thread % 32) // width == 2
        if selected or not divergent:
            payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = inputs[thread * _ITEMS + item]
            for iteration in range(iterations):
                payload = api.exchange(group, payload, mode="blocked_to_striped")
                for item in cutlass.range_constexpr(_ITEMS):
                    payload[item] = payload[item] + iteration
            for item in cutlass.range_constexpr(_ITEMS):
                outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        kernel(source, observed, iterations).launch(grid=1, block=_BLOCK)

    iterations = 5
    source = values_for(np.int32, _TILE, shift=37)
    observed = np.full_like(source, -101)
    transformed = source.copy()
    for iteration in range(iterations):
        transformed = _layout(transformed, width, "blocked_to_striped") + iteration
    expected = observed.copy()
    for first in range(0, _THREADS, width):
        selected = (
            True
            if width == 64
            else first // 32 == 1
            if width == 32
            else (first % 32) // width == 2
        )
        if selected or not divergent:
            expected[first * _ITEMS : (first + width) * _ITEMS] = transformed[
                first * _ITEMS : (first + width) * _ITEMS
            ]
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("warp", (False, True), ids=("block", "logical-warp"))
def test_final_cubin(tmp_path, warp):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        if cutlass.const_expr(warp):
            group = cutlass_coop.this_warp().group_by(8)
        else:
            group = cutlass_coop.this_block()
        payload = cutlass_coop.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        result = cutlass_coop.exchange(group, payload, mode="blocked_to_striped")
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = result[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _TILE, shift=43)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, out)
        compiled(src, out)
    np.testing.assert_array_equal(
        observed, _layout(source, 8 if warp else _THREADS, "blocked_to_striped")
    )
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
        assert "cuda_coop_cutlass_" not in sass
        assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
        if warp:
            assert re.search(r"\bBAR(?:\.[A-Z0-9_]+)*\b", sass) is None
