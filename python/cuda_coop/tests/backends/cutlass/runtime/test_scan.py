# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check Scan values, types, aggregates, and which lanes may use results.

NumPy prefix folds provide an independent oracle with the input dtype.
Payload cases also check that Scan preserves its input. Partial warp scans
consume primary results only inside their valid prefix, while aggregates
remain available to all group members and exclude the exclusive seed.
"""

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
_FORMS = (
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "exclusive_sum",
    "inclusive_sum",
)
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
    """Build the host prefix oracle without widening the accumulator dtype.

    Inclusive mode folds the input directly. Exclusive mode starts with the
    seed and omits the final input, giving one result per original element.
    """

    if inclusive:
        return _UFUNCS[operation].accumulate(values, dtype=values.dtype)
    initial = np.array([seed], dtype=values.dtype)
    return _UFUNCS[operation].accumulate(
        np.concatenate((initial, values[:-1])), dtype=values.dtype
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize(
    "items_per_thread", (0, 1, 4), ids=("scalar", "one-item", "payload")
)
def test_spellings_types(api, dtype, items_per_thread):
    """Compare every public Scan spelling while preserving the original input.

    The same scalar or payload passes through all five entry points. Results
    must retain scalar type or payload extent and minimum alignment. Copying
    the input after the calls checks that none of them modifies it.
    """

    value_type = cutlass_dtype(dtype)
    extent = max(items_per_thread, 1)
    size = _THREADS * extent
    functions = tuple(getattr(api, name) for name in _FORMS)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
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
        if cutlass.const_expr(items_per_thread):
            value = api.ThreadData(
                items_per_thread, dtype=value_type, alignment=64
            )
            for item in cutlass.range_constexpr(items_per_thread):
                value[item] = inputs[thread * items_per_thread + item]
        else:
            value = inputs[thread]
        for case in cutlass.range_constexpr(5):
            result = functions[case](api.this_block(), value)
            if cutlass.const_expr(items_per_thread):
                assert result.items_per_thread == items_per_thread
                assert result.alignment >= 64
                for item in cutlass.range_constexpr(items_per_thread):
                    outputs[case * size + thread * items_per_thread + item] = (
                        result[item]
                    )
            else:
                assert isinstance(result, value_type)
                outputs[case * size + thread] = result
        if cutlass.const_expr(items_per_thread):
            for item in cutlass.range_constexpr(items_per_thread):
                checks[thread * items_per_thread + item] = value[item]
        else:
            checks[thread] = value

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, preserved, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = (np.arange(size) % 3).astype(dtype)
    if np.dtype(dtype).kind != "u":
        source -= 1
    observed = np.zeros(5 * size, dtype=dtype)
    preserved = np.zeros_like(source)
    expected = np.stack(
        [
            _prefix(source, inclusive=name.startswith("inclusive"))
            for name in _FORMS
        ]
    )
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check, items_per_thread)
    np.testing.assert_array_equal(observed.reshape(5, size), expected)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("runtime", (False, True), ids=("literal", "typed"))
def test_initial_type(api, dtype, runtime):
    """Use an exclusive seed as a Python literal or typed runtime operand.

    Each input dtype must produce its own scalar result type. Comparing both
    paths with the same prefix oracle exposes seed conversion differences.
    """

    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(
        source: cute.Pointer, observed: cute.Pointer, initial: value_type
    ):
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
    def launch(
        source: cute.Pointer, observed: cute.Pointer, initial: value_type
    ):
        kernel(source, observed, initial).launch(grid=1, block=_THREADS)

    source = (np.arange(_THREADS) % 2).astype(dtype)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, 7)
    np.testing.assert_array_equal(
        observed, _prefix(source, inclusive=False, seed=7)
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", (np.int32, np.float32))
@pytest.mark.parametrize("items_per_thread", (0, 2), ids=("scalar", "payload"))
@pytest.mark.parametrize(
    "numpy_seed", (False, True), ids=("cute-seed", "numpy-seed")
)
def test_numpy_input(api, dtype, items_per_thread, numpy_seed):
    """Convert NumPy scalar values and payload items into typed Scan operands.

    The seed comes either from NumPy or the matching CuTe scalar constructor.
    Scanning ones makes every expected position explicit, so both conversion
    paths must produce the same sequence starting at seven.
    """

    value_type = cutlass_dtype(dtype)
    size = _THREADS * max(items_per_thread, 1)

    @cute.kernel
    def kernel(observed: cute.Pointer, items_per_thread: cutlass.Constexpr):
        thread = cute.arch.thread_idx()[0]
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        if cutlass.const_expr(items_per_thread):
            value = api.ThreadData(items_per_thread, dtype=dtype)
            for item in cutlass.range_constexpr(items_per_thread):
                value[item] = dtype(1)
        else:
            value = dtype(1)
        if cutlass.const_expr(numpy_seed):
            result = api.scan(
                api.this_block(),
                value,
                mode="exclusive",
                initial_value=dtype(7),
            )
        else:
            result = api.exclusive_scan(
                api.this_block(), value, initial_value=value_type(7)
            )
        if cutlass.const_expr(items_per_thread):
            for item in cutlass.range_constexpr(items_per_thread):
                outputs[thread * items_per_thread + item] = result[item]
        else:
            outputs[thread] = result

    @cute.jit
    def launch(
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(observed, items_per_thread).launch(grid=1, block=_THREADS)

    observed = np.zeros(size, dtype=dtype)
    with device_array(observed) as out:
        launch(out, items_per_thread)
    np.testing.assert_array_equal(observed, np.arange(size, dtype=dtype) + 7)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("operation", tuple(_UFUNCS))
@pytest.mark.parametrize(
    "inclusive", (False, True), ids=("exclusive", "inclusive")
)
def test_builtins(api, operation, inclusive):
    """Fold two items per thread in blocked order for every built-in operator.

    Exclusive cases use an operator-specific runtime seed; inclusive cases
    start from the input. Products use positive and negative ones so the test
    can check ordering and seed behavior without large intermediate values.
    """

    seed = _SEEDS[operation]
    size = 2 * _THREADS

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        initial: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        payload = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(2):
            payload[item] = inputs[thread * 2 + item]
        if cutlass.const_expr(inclusive):
            result = api.scan(
                api.this_block(), payload, mode="inclusive", scan_op=operation
            )
        else:
            result = api.exclusive_scan(
                api.this_block(),
                payload,
                scan_op=operation,
                initial_value=initial,
            )
        for item in cutlass.range_constexpr(2):
            outputs[thread * 2 + item] = result[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        initial: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, initial, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

    source = values_for(np.int32, size, shift=17)
    if operation == "multiplies":
        source[:] = 1
        source[::7] = -1
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, seed, 2)
    np.testing.assert_array_equal(
        observed,
        _prefix(source, inclusive=inclusive, operation=operation, seed=seed),
    )


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("items_per_thread", (0, 3), ids=("scalar", "payload"))
def test_block_algorithm(api, algorithm, items_per_thread):
    extent = max(items_per_thread, 1)
    size = _THREADS * extent

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(size))
        outputs = cute.make_tensor(observed, cute.make_layout(size))
        if cutlass.const_expr(items_per_thread):
            value = api.ThreadData(items_per_thread, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(items_per_thread):
                value[item] = inputs[thread * items_per_thread + item]
        else:
            value = inputs[thread]
        result = api.inclusive_sum(api.this_block(), value, algorithm=algorithm)
        if cutlass.const_expr(items_per_thread):
            for item in cutlass.range_constexpr(items_per_thread):
                outputs[thread * items_per_thread + item] = result[item]
        else:
            outputs[thread] = result

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, items_per_thread).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, size, shift=23)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, items_per_thread)
    np.testing.assert_array_equal(observed, source.cumsum(dtype=np.int32))


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize(
    "width",
    (None, *_WIDTHS),
    ids=(
        "physical",
        "width1",
        "width2",
        "width4",
        "width8",
        "width16",
        "width32",
    ),
)
def test_warp_forms(api, width):
    """Check that each Scan spelling restarts at its physical or logical warp.

    The host splits the input into independent rows of the selected width.
    Comparing all five forms with those row prefixes catches results carried
    across a group boundary.
    """

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
            outputs[case * _THREADS + thread] = functions[case](
                group, inputs[thread]
            )

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
    """Check valid prefixes and the aggregate available to all members.

    Only lanes below the prefix count write their primary result. All members
    record the aggregate, which folds valid inputs without the exclusive seed.
    Static and runtime counts must preserve these two different output rules.
    """

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
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        totals = cute.make_tensor(aggregates, cute.make_layout(_THREADS))
        aggregate = cutlass_coop.ThreadData(items_per_thread, alignment=64)
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
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(
            source, observed, aggregates, valid, initial, items_per_thread
        ).launch(grid=1, block=_BLOCK)

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
        launch(src, out, agg, count, seed, 1)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(aggregates, totals)


@pytest.mark.parametrize("operation", tuple(_UFUNCS))
def test_block_aggregate(operation):
    """Keep the exclusive seed out of the block aggregate for every operator.

    Every member records its seeded prefix and the block-wide input fold.
    The host computes the aggregate from the input alone. It uses the seed
    only when building the expected exclusive prefixes.
    """

    seed = _SEEDS[operation]

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        totals = cute.make_tensor(aggregates, cute.make_layout(_THREADS))
        aggregate = cutlass_coop.ThreadData(
            items_per_thread, dtype=cutlass.Int32
        )
        outputs[thread] = cutlass_coop.exclusive_scan(
            cutlass_coop.this_block(),
            inputs[thread],
            scan_op=operation,
            initial_value=seed,
            aggregate_output=aggregate,
        )
        totals[thread] = aggregate[0]

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, observed, aggregates, items_per_thread).launch(
            grid=1, block=_BLOCK
        )

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
        launch(src, out, agg, 1)
    np.testing.assert_array_equal(
        observed,
        _prefix(source, inclusive=False, operation=operation, seed=seed),
    )
    np.testing.assert_array_equal(
        aggregates, np.full_like(source, _UFUNCS[operation].reduce(source))
    )


@pytest.mark.parametrize("value", (-1, 0, 9, (1 << 32) + 1))
def test_bad_prefix(tmp_path, value):
    """Check runtime prefix traps in a separate CUDA process for each case.

    A device trap can leave its context unusable. The child confirms that it
    imports the same package source as the parent, then launches with an Int64
    count and reports a CUDA trap status. A value above 32 bits would look
    valid if truncated, so it checks that validation sees the full count. An
    unrelated child-process failure is not enough to satisfy the test.
    """

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
""")  # noqa: E501 - Preserve embedded source bytes.
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
        check=False,
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
    """Check Scan results, aggregates, and preservation of register inputs.

    Both a register tensor and the TensorSSA value loaded from it use the
    qualified API. Separate outputs check the seeded prefix, the unseeded
    aggregate at every thread, and the original register contents after Scan.
    """

    size = 2 * _THREADS

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        aggregates: cute.Pointer,
        preserved: cute.Pointer,
        items_per_thread: cutlass.Constexpr,
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
        aggregate = cutlass_coop.ThreadData(items_per_thread)
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
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(
            source, observed, aggregates, preserved, items_per_thread
        ).launch(grid=1, block=_THREADS)

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
        launch(src, out, agg, check, 1)
    np.testing.assert_array_equal(
        observed, _prefix(source, inclusive=False, seed=7)
    )
    np.testing.assert_array_equal(
        aggregates, np.full_like(aggregates, source.sum(dtype=np.int32))
    )
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("warp", (False, True), ids=("block", "logical-warp"))
def test_final_cubin(tmp_path, warp):
    """Check Scan inlining and the absence of block barriers in warp kernels.

    After checking the prefix values, inspect the linked cubin. A provider
    symbol or CALL instruction in the machine code means the CUB wrapper was
    not inlined. The logical-warp path must contain no block barrier (BAR).
    Resource reports are saved for inspection but have no assertions here.
    """

    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip(
            "cuobjdump is required to inspect final linked instructions"
        )

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
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](
            launch, src, out
        )
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
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_example(api, items_per_thread):
    path = PACKAGE_ROOT / "examples/cutlass/scan.py"
    spec = importlib.util.spec_from_file_location("cutlass_scan_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api, items_per_thread=items_per_thread)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", ("raking", "raking_memoize"))
def test_partial_block(api, algorithm):
    """Scan a complete block that ends in a partial physical warp.

    The raking algorithms support all 48 threads of this block. This differs
    from the warp_scans and logical-warp routes, whose launch checks require
    complete physical warps.
    """

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
