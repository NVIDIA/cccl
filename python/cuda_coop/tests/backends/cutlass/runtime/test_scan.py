# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent prefix, aggregate, and payload-preservation Scan oracles."""

import importlib.util
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from cuda import coop
from cuda.bindings import driver
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
_BLOCK = (8, 4, 2)
_THREADS = 64
_FORMS = ("scan", "exclusive_scan", "inclusive_scan", "exclusive_sum", "inclusive_sum")
_WIDTHS = (1, 2, 4, 8, 16, 32)
_ALGORITHMS = ("raking", "raking_memoize", "warp_scans")
_UFUNCS = {
    "sum": np.add,
    "multiplies": np.multiply,
    "min": np.minimum,
    "max": np.maximum,
    "bit_and": np.bitwise_and,
    "bit_or": np.bitwise_or,
    "bit_xor": np.bitwise_xor,
}
_SEEDS = {
    "sum": 7,
    "multiplies": 3,
    "min": 19,
    "max": -19,
    "bit_and": 127,
    "bit_or": 64,
    "bit_xor": 85,
}


def _prefix(values, *, inclusive, operation="sum", seed=0):
    if inclusive:
        return _UFUNCS[operation].accumulate(values, dtype=values.dtype)
    initial = np.array([seed], dtype=values.dtype)
    return _UFUNCS[operation].accumulate(
        np.concatenate((initial, values[:-1])), dtype=values.dtype
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("items", (0, 1, 2), ids=("scalar", "one-item", "payload"))
def test_spellings_types(api, dtype, items):
    value_type = cutlass_dtype(dtype)
    extent = max(items, 1)
    size = _THREADS * extent
    functions = tuple(getattr(api, name) for name in _FORMS)

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(5 * size)), value_type
        )
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), value_type
        )
        if cutlass.const_expr(items):
            value = api.ThreadData(items, dtype=value_type, alignment=64)
            for item in cutlass.range_constexpr(items):
                value[item] = inputs[thread * items + item]
        else:
            value = inputs[thread]
        for case in cutlass.range_constexpr(5):
            result = functions[case](api.this_block(), value)
            if cutlass.const_expr(items):
                assert result.items_per_thread == items
                assert result.alignment >= 64
                for item in cutlass.range_constexpr(items):
                    outputs[case * size + thread * items + item] = result[item]
            else:
                assert isinstance(result, value_type)
                outputs[case * size + thread] = result
        if cutlass.const_expr(items):
            for item in cutlass.range_constexpr(items):
                checks[thread * items + item] = value[item]
        else:
            checks[thread] = value

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        kernel(source, observed, preserved).launch(grid=1, block=_BLOCK)

    source = (np.arange(size) % 3).astype(dtype)
    if np.dtype(dtype).kind != "u":
        source -= 1
    observed = np.zeros(5 * size, dtype=dtype)
    preserved = np.zeros_like(source)
    expected = np.stack(
        [_prefix(source, inclusive=name.startswith("inclusive")) for name in _FORMS]
    )
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check)
    np.testing.assert_array_equal(observed.reshape(5, size), expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("runtime", (False, True), ids=("literal", "typed"))
def test_initial_type(api, dtype, runtime):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, initial: value_type):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_THREADS)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(_THREADS)), value_type
        )
        if cutlass.const_expr(runtime):
            result = api.exclusive_scan(
                api.this_block(), inputs[thread], initial_value=initial
            )
        else:
            result = api.exclusive_scan(
                api.this_block(), inputs[thread], initial_value=7
            )
        assert isinstance(result, value_type)
        outputs[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, initial: value_type):
        kernel(source, observed, initial).launch(grid=1, block=_THREADS)

    source = (np.arange(_THREADS) % 2).astype(dtype)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, 7)
    np.testing.assert_array_equal(observed, _prefix(source, inclusive=False, seed=7))


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("operation", tuple(_UFUNCS))
@pytest.mark.parametrize("inclusive", (False, True), ids=("exclusive", "inclusive"))
def test_builtins(api, operation, inclusive):
    seed = _SEEDS[operation]
    size = 2 * _THREADS

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, initial: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        payload = api.ThreadData(2, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(2):
            payload[item] = inputs[thread * 2 + item]
        if cutlass.const_expr(inclusive):
            result = api.scan(
                api.this_block(), payload, mode="inclusive", scan_op=operation
            )
        else:
            result = api.exclusive_scan(
                api.this_block(), payload, scan_op=operation, initial_value=initial
            )
        for item in cutlass.range_constexpr(2):
            outputs[thread * 2 + item] = result[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, initial: cutlass.Int32):
        kernel(source, observed, initial).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, size, shift=17)
    if operation == "multiplies":
        source[:] = 1
        source[::7] = -1
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, seed)
    np.testing.assert_array_equal(
        observed, _prefix(source, inclusive=inclusive, operation=operation, seed=seed)
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("items", (0, 3), ids=("scalar", "payload"))
def test_block_algorithm(api, algorithm, items):
    extent = max(items, 1)
    size = _THREADS * extent

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        if cutlass.const_expr(items):
            value = api.ThreadData(items, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(items):
                value[item] = inputs[thread * items + item]
        else:
            value = inputs[thread]
        result = api.inclusive_sum(api.this_block(), value, algorithm=algorithm)
        if cutlass.const_expr(items):
            for item in cutlass.range_constexpr(items):
                outputs[thread * items + item] = result[item]
        else:
            outputs[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, size, shift=23)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, source.cumsum(dtype=np.int32))


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize(
    "width",
    (None, *_WIDTHS),
    ids=("physical", "width1", "width2", "width4", "width8", "width16", "width32"),
)
def test_warp_forms(api, width):
    group_width = width or 32
    functions = tuple(getattr(api, name) for name in _FORMS)

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(5 * _THREADS))
        group = api.this_warp()
        if cutlass.const_expr(width is not None):
            group = group.group_by(width)
        for case in cutlass.range_constexpr(5):
            outputs[case * _THREADS + thread] = functions[case](group, inputs[thread])

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS, shift=31)
    observed = np.zeros(5 * _THREADS, dtype=np.int32)
    expected = np.stack(
        [
            np.concatenate(
                [
                    _prefix(row, inclusive=name.startswith("inclusive"))
                    for row in source.reshape(-1, group_width)
                ]
            )
            for name in _FORMS
        ]
    )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed.reshape(5, _THREADS), expected)


@pytest.mark.parametrize("width", (1, 8, 32))
@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("runtime", (False, True), ids=("static", "runtime"))
def test_prefix_aggregate(width, form, runtime):
    count = max(1, width - 3)
    function = getattr(cutlass_coop, form)
    seeded = form in {"scan", "exclusive_scan"}
    seed = 11 if seeded else 0

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        valid: cutlass.Int64,
        initial: cutlass.Int32,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        totals = cute.make_tensor(aggregates, cute.make_layout(_THREADS))
        aggregate = cutlass_coop.ThreadData(1, alignment=64)
        group = cutlass_coop.this_warp().group_by(width)
        if cutlass.const_expr(runtime):
            valid_items = valid
        else:
            valid_items = count
        if cutlass.const_expr(seeded):
            result = function(
                group,
                inputs[thread],
                initial_value=initial,
                valid_items=valid_items,
                aggregate_output=aggregate,
            )
        else:
            result = function(
                group,
                inputs[thread],
                valid_items=valid_items,
                aggregate_output=aggregate,
            )
        if thread % width < count:
            outputs[thread] = result
        totals[thread] = aggregate[0]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        valid: cutlass.Int64,
        initial: cutlass.Int32,
    ):
        kernel(source, observed, aggregates, valid, initial).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, _THREADS, shift=37)
    observed = np.full_like(source, -101)
    aggregates = np.zeros_like(source)
    expected = observed.copy()
    totals = np.zeros_like(source)
    for start in range(0, _THREADS, width):
        values = source[start : start + count]
        expected[start : start + count] = _prefix(
            values, inclusive=form.startswith("inclusive"), seed=seed
        )
        totals[start : start + width] = values.sum(dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(aggregates) as agg,
    ):
        launch(src, out, agg, count, seed)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(aggregates, totals)


@pytest.mark.parametrize("operation", tuple(_UFUNCS))
def test_block_aggregate(operation):
    seed = _SEEDS[operation]

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, aggregates: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        totals = cute.make_tensor(aggregates, cute.make_layout(_THREADS))
        aggregate = cutlass_coop.ThreadData(1, dtype=cutlass.Int32)
        outputs[thread] = cutlass_coop.exclusive_scan(
            cutlass_coop.this_block(),
            inputs[thread],
            scan_op=operation,
            initial_value=seed,
            aggregate_output=aggregate,
        )
        totals[thread] = aggregate[0]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, aggregates: cute.Pointer):
        kernel(source, observed, aggregates).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS, shift=41)
    if operation == "multiplies":
        source[:] = 1
        source[::7] = -1
    observed = np.zeros_like(source)
    aggregates = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(aggregates) as agg,
    ):
        launch(src, out, agg)
    np.testing.assert_array_equal(
        observed, _prefix(source, inclusive=False, operation=operation, seed=seed)
    )
    np.testing.assert_array_equal(
        aggregates, np.full_like(source, _UFUNCS[operation].reduce(source))
    )


@pytest.mark.parametrize("value", (-1, 0, 9, (1 << 32) + 1))
def test_bad_prefix(tmp_path, value):
    path = tmp_path / "invalid_prefix.py"
    path.write_text(f"""import numpy as np
import cutlass
from cutlass import cute
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array

assert cutlass_coop.__file__ == {cutlass_coop.__file__!r}

@cute.kernel
def kernel(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int64):
    thread = cute.arch.thread_idx()[0]
    inputs = cute.make_tensor(source, cute.make_layout(64))
    outputs = cute.make_tensor(observed, cute.make_layout(64))
    outputs[thread] = cutlass_coop.exclusive_sum(cutlass_coop.this_warp().group_by(8), inputs[thread], valid_items=count)

@cute.jit
def launch(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int64):
    kernel(source, observed, count).launch(grid=1, block=64)

with device_array(np.ones(64, dtype=np.int32)) as src, device_array(np.zeros(64, dtype=np.int32)) as out:
    launch(src, out, {value})
    status = driver.cuCtxSynchronize()[0]
    print(f"prefix launch status: {{int(status)}}", flush=True)
raise AssertionError("invalid Scan prefix did not trap")
""")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, str(path)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert any(
        f"prefix launch status: {int(status)}" in output
        for status in (
            driver.CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION,
            driver.CUresult.CUDA_ERROR_LAUNCH_FAILED,
        )
    ), output


@pytest.mark.parametrize("ssa", (False, True), ids=("rmem", "tensor-ssa"))
def test_register_aggregate(ssa):
    size = 2 * _THREADS

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        preserved: cute.Pointer,
    ):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        totals = cute.make_tensor(aggregates, cute.make_layout(_THREADS))
        checks = cute.make_tensor(preserved, cute.make_layout(size))
        fragment = cute.make_rmem_tensor((2,), cutlass.Int32)
        for item in cutlass.range_constexpr(2):
            fragment[item] = inputs[thread * 2 + item]
        if cutlass.const_expr(ssa):
            value = fragment.load()
        else:
            value = fragment
        aggregate = cutlass_coop.ThreadData(1)
        result = cutlass_coop.exclusive_scan(
            cutlass_coop.this_block(),
            value,
            initial_value=7,
            aggregate_output=aggregate,
        )
        for item in cutlass.range_constexpr(2):
            outputs[thread * 2 + item] = result[item]
            checks[thread * 2 + item] = fragment[item]
        totals[thread] = aggregate[0]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        preserved: cute.Pointer,
    ):
        kernel(source, observed, aggregates, preserved).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, size, shift=47)
    observed = np.zeros_like(source)
    aggregates = np.zeros(_THREADS, dtype=np.int32)
    preserved = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(aggregates) as agg,
        device_array(preserved) as check,
    ):
        launch(src, out, agg, check)
    np.testing.assert_array_equal(observed, _prefix(source, inclusive=False, seed=7))
    np.testing.assert_array_equal(
        aggregates, np.full_like(aggregates, source.sum(dtype=np.int32))
    )
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("warp", (False, True), ids=("block", "logical-warp"))
def test_final_cubin(tmp_path, warp):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        if cutlass.const_expr(warp):
            group = cutlass_coop.this_warp().group_by(8)
        else:
            group = cutlass_coop.this_block()
        outputs[thread] = cutlass_coop.inclusive_sum(group, inputs[thread])

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _THREADS, shift=43)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, out)
        compiled(src, out)
    width = 8 if warp else _THREADS
    expected = np.concatenate(
        [row.cumsum(dtype=np.int32) for row in source.reshape(-1, width)]
    )
    np.testing.assert_array_equal(observed, expected)
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


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/scan.py"
    spec = importlib.util.spec_from_file_location("cutlass_scan_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", ("raking", "raking_memoize"))
def test_partial_block(api, algorithm):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + 8 * (y + 3 * z)
        inputs = cute.make_tensor(source, cute.make_layout(48))
        outputs = cute.make_tensor(observed, cute.make_layout(48))
        outputs[thread] = api.inclusive_sum(
            api.this_block(), inputs[thread], algorithm=algorithm
        )

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=(8, 3, 2))

    source = values_for(np.int64, 48, shift=53)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, source.cumsum(dtype=np.int64))
