# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run scalar CuTe callbacks through CUB and inspect final linked kernels."""

import re
import shutil
import subprocess

import cutlass
import numpy as np
import pytest
from cutlass import cute
from cutlass.base_dsl.common import get_current_env_manager
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

import cuda.coop.cutlass as coop
from tests.runtime_support import cutlass_dtype, device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_THREADS = 64
_SEED = -7


def _callback(offset=1, dtype=cutlass.Int32):
    @cute.jit
    def callback(a, b) -> dtype:
        return a + b + offset

    return callback


def _launcher(operation, callback, *, warp=False, nested=False):
    value_type = callback.dtype if callback is not None else cutlass.Int32

    @cute.jit
    def collective(group, value):
        if cutlass.const_expr(operation == "reduce"):
            return coop.reduce(group, value, binary_op=callback)
        elif cutlass.const_expr(operation == "inclusive"):
            return coop.inclusive_scan(group, value, scan_op=callback)
        else:
            return coop.exclusive_scan(
                group, value, scan_op=callback, initial_value=_SEED
            )

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        source_tensor = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_THREADS)), value_type
        )
        output = cute.recast_tensor(
            cute.make_tensor(destination, cute.make_layout(_THREADS)),
            value_type,
        )
        if cutlass.const_expr(warp):
            group = coop.this_warp().group_by(8)
        else:
            group = coop.this_block()
        value = source_tensor[thread]
        if cutlass.const_expr(nested):
            result = collective(group, value)
        elif cutlass.const_expr(operation == "reduce"):
            result = coop.reduce(group, value, binary_op=callback)
        elif cutlass.const_expr(operation == "inclusive"):
            result = coop.inclusive_scan(group, value, scan_op=callback)
        else:
            result = coop.exclusive_scan(
                group, value, scan_op=callback, initial_value=_SEED
            )
        if cutlass.const_expr(operation == "reduce"):
            if group.rank() == 0:
                output[thread] = result
        else:
            output[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_THREADS)

    return launch


def _expected(source, operation, *, warp=False, offset=1):
    width = 8 if warp else _THREADS
    rows = source.reshape(-1, width)
    result = np.full_like(source, -101).reshape(-1, width)
    if operation == "reduce":
        result[:, 0] = rows.sum(axis=1, dtype=source.dtype) + offset * (
            width - 1
        )
    elif operation == "inclusive":
        result[:] = rows.cumsum(
            axis=1, dtype=source.dtype
        ) + offset * np.arange(width)
    else:
        result[:, 0] = _SEED
        result[:, 1:] = (
            _SEED
            + rows[:, :-1].cumsum(axis=1, dtype=source.dtype)
            + offset * np.arange(1, width)
        )
    return result.reshape(-1)


def _run(launch, source, *, options=()):
    observed = np.full_like(source, 0 if source.dtype.kind == "u" else -101)
    with device_array(source) as src, device_array(observed) as dst:
        compile_fn = cute.compile[options] if options else cute.compile
        compiled = compile_fn(launch, src, dst)
        assert get_current_env_manager() is None
        compiled(src, dst)
    return observed


@pytest.mark.parametrize("dtype", (np.int32, np.float32))
@pytest.mark.parametrize("warp", (False, True), ids=("block", "logical-warp"))
@pytest.mark.parametrize("operation", ("reduce", "inclusive", "exclusive"))
def test_scalar_callback_values(dtype, warp, operation):
    scalar_type = cutlass_dtype(dtype)
    callback = coop.compile_operator(_callback(dtype=scalar_type), scalar_type)
    source = values_for(dtype, _THREADS, shift=19)
    observed = _run(_launcher(operation, callback, warp=warp), source)
    np.testing.assert_array_equal(
        observed, _expected(source, operation, warp=warp)
    )


@pytest.mark.parametrize(
    "dtype",
    (
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.uint32,
        np.int64,
        np.uint64,
        np.float64,
    ),
)
def test_scalar_callback_abi(dtype):
    """Preserve narrow, unsigned and wide values through the native call ABI."""

    scalar_type = cutlass_dtype(dtype)

    @cute.jit
    def maximum(a, b) -> scalar_type:
        return a if a > b else b

    callback = coop.compile_operator(maximum, scalar_type)
    source = values_for(dtype, _THREADS, shift=37)
    observed = _run(_launcher("reduce", callback, warp=True), source)
    expected = np.full_like(source, 0 if source.dtype.kind == "u" else -101)
    expected[::8] = source.reshape(-1, 8).max(axis=1)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("same_callable", (False, True))
def test_same_name_and_signature_keep_distinct_bodies(same_callable):
    """Use both callbacks in one kernel, then repeat separate compilations."""

    if same_callable:
        offset = 1

        @cute.jit
        def callback(a, b) -> cutlass.Int32:
            return a + b + offset

        first = coop.compile_operator(callback, cutlass.Int32)
        offset = 3
        second = coop.compile_operator(callback, cutlass.Int32)
        # Prepared operators must retain their original compiled arithmetic.
        offset = 9
    else:
        first = coop.compile_operator(_callback(1), cutlass.Int32)
        second = coop.compile_operator(_callback(3), cutlass.Int32)
    assert first.symbol != second.symbol

    @cute.kernel
    def kernel(source: cute.Pointer, destination: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        values = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(destination, cute.make_layout(2))
        group = coop.this_block()
        left = coop.reduce(group, values[thread], binary_op=first)
        right = coop.reduce(group, values[thread], binary_op=second)
        if thread == 0:
            outputs[0] = left
            outputs[1] = right

    @cute.jit
    def launch(source: cute.Pointer, destination: cute.Pointer):
        kernel(source, destination).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _THREADS, shift=31)
    observed = np.full(2, -101, dtype=np.int32)
    with device_array(source) as src, device_array(observed) as dst:
        compiled = cute.compile(launch, src, dst)
        compiled(src, dst)
    np.testing.assert_array_equal(
        observed, source.sum(dtype=np.int32) + np.array([1, 3]) * (_THREADS - 1)
    )
    for callback, offset in ((first, 1), (second, 3), (first, 1)):
        observed = _run(_launcher("reduce", callback), source)
        np.testing.assert_array_equal(
            observed, _expected(source, "reduce", offset=offset)
        )
    observed = _run(_launcher("reduce", None), source)
    np.testing.assert_array_equal(
        observed, _expected(source, "reduce", offset=0)
    )


@pytest.mark.parametrize("nested", (False, True), ids=("direct", "nested-jit"))
def test_repeated_compile_and_nested_calls(nested):
    callback = coop.compile_operator(_callback(), cutlass.Int32)
    launch = _launcher("inclusive", callback, nested=nested)
    for shift in (17, 43):
        source = values_for(np.int32, _THREADS, shift=shift)
        np.testing.assert_array_equal(
            _run(launch, source), _expected(source, "inclusive")
        )


def test_failed_trace_after_callback_registration_can_retry():
    callback = coop.compile_operator(_callback(), cutlass.Int32)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        destination: cute.Pointer,
        reject: cutlass.Constexpr,
    ):
        thread = cute.arch.thread_idx()[0]
        values = cute.make_tensor(source, cute.make_layout(_THREADS))
        result = coop.reduce(
            coop.this_block(), values[thread], binary_op=callback
        )
        if cutlass.const_expr(reject):
            raise ValueError("injected failure after callback registration")
        if thread == 0:
            cute.make_tensor(destination, cute.make_layout(_THREADS))[
                thread
            ] = result

    @cute.jit
    def launch(
        source: cute.Pointer,
        destination: cute.Pointer,
        reject: cutlass.Constexpr,
    ):
        kernel(source, destination, reject).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _THREADS, shift=23)
    observed = np.full_like(source, -101)
    with device_array(source) as src, device_array(observed) as dst:
        with pytest.raises(ValueError, match="after callback registration"):
            cute.compile(launch, src, dst, True)
        assert get_current_env_manager() is None
        compiled = cute.compile(launch, src, dst, False)
        compiled(src, dst)
    np.testing.assert_array_equal(observed, _expected(source, "reduce"))


@pytest.mark.parametrize("operation", ("reduce", "inclusive"))
def test_final_cubin_inlines_provider_and_callback(tmp_path, operation):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip(
            "cuobjdump is required to inspect final linked instructions"
        )
    callback = coop.compile_operator(_callback(), cutlass.Int32)
    source = values_for(np.int32, _THREADS, shift=47)
    observed = _run(
        _launcher(operation, callback),
        source,
        options=(KeepCUBIN, DumpDir(str(tmp_path))),
    )
    np.testing.assert_array_equal(observed, _expected(source, operation))
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins, "CUTLASS did not retain the final linked cubin"
    for cubin in cubins:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        cubin.with_suffix(".sass").write_text(sass)
        assert "cuda_coop_cutlass_" not in sass
        assert callback.symbol not in sass
        assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
