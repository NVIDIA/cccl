# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Independent numerical and ownership checks for built-in Reduce and Sum."""

import importlib.util
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
    check_cuda,
    cutlass_dtype,
    device_array,
    values_for,
)
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_APIS = (coop, cutlass_coop)
_BLOCK = (8, 4, 4)
_THREADS = 128
_WIDTHS = {"thread": 1, "warp": 32, "logical": 8, "block": _THREADS, "mapped": 64}
_OPS = ("sum", "multiplies", "min", "max", "bit_and", "bit_or", "bit_xor")
_ALGORITHMS = ("raking_commutative_only", "raking", "warp_reductions")


def _group(api, kind):
    if kind == "thread":
        return api.this_thread()
    if kind == "warp":
        return api.this_warp()
    if kind == "logical":
        return api.this_warp().group_by(8)
    if kind == "mapped":
        return api.this_block().group_by(2)
    return api.this_block()


def _fold(values, operation):
    operation = {
        "sum": np.add,
        "multiplies": np.multiply,
        "min": np.minimum,
        "max": np.maximum,
        "bit_and": np.bitwise_and,
        "bit_or": np.bitwise_or,
        "bit_xor": np.bitwise_xor,
    }[operation]
    return operation.reduce(values.reshape(-1), dtype=values.dtype)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("items", (0, 1, 2), ids=("scalar", "one-item", "two-items"))
def test_sum_types(api, dtype, items):
    value_type = cutlass_dtype(dtype)
    extent = max(items, 1)
    size = _THREADS * extent

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(size)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(_THREADS)), value_type
        )
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(size)), value_type
        )
        if cutlass.const_expr(items == 0):
            value = inputs[thread]
            result = api.sum(api.this_block(), value)
            checks[thread] = value
        else:
            payload = api.ThreadData(items, dtype=value_type)
            for item in cutlass.range_constexpr(items):
                payload[item] = inputs[thread * items + item]
            result = api.sum(api.this_block(), payload)
            for item in cutlass.range_constexpr(items):
                checks[thread * items + item] = payload[item]
        assert isinstance(result, value_type)
        outputs[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        kernel(source, observed, preserved).launch(grid=1, block=_BLOCK)

    base = np.arange(size) % 3
    if np.dtype(dtype).kind != "u":
        base -= 1
    source = base.astype(dtype)
    observed = np.zeros(_THREADS, dtype=dtype)
    preserved = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check)
    np.testing.assert_array_equal(
        observed, np.full_like(observed, _fold(source, "sum"))
    )
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("kind", tuple(_WIDTHS))
@pytest.mark.parametrize("operation", _OPS)
@pytest.mark.parametrize("broadcast", (False, True), ids=("root", "members"))
def test_group_builtins(api, kind, operation, broadcast):
    width = _WIDTHS[kind]
    items = 2

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS * items))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        payload = api.ThreadData(items, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(items):
            payload[item] = inputs[thread * items + item]
        result = api.reduce(
            _group(api, kind), payload, binary_op=operation, broadcast=broadcast
        )
        if broadcast or thread % width == 0:
            outputs[thread] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS * items, shift=11)
    if operation == "multiplies":
        source[:] = 1
        source[::7] = -1
    observed = np.full(_THREADS, -101, dtype=np.int32)
    expected = observed.copy()
    for start in range(0, _THREADS, width):
        result = _fold(source[start * items : (start + width) * items], operation)
        expected[start : start + width if broadcast else start + 1] = result
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("dtype", (np.uint8, np.uint16, np.uint32, np.uint64))
def test_unsigned_result(api, dtype):
    value_type = cutlass_dtype(dtype)

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_THREADS)), value_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(_THREADS)), value_type
        )
        first = api.reduce(api.this_warp(), inputs[thread], binary_op="max")
        assert isinstance(first, value_type)
        second = api.reduce(api.this_block(), first, binary_op="max")
        assert isinstance(second, value_type)
        payload = api.ThreadData(1, dtype=value_type)
        payload[0] = second
        outputs[thread] = payload[0]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_BLOCK)

    source = values_for(dtype, _THREADS, shift=17)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    np.testing.assert_array_equal(observed, np.full_like(source, source.max()))


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("kind", ("block", "warp", "logical"))
@pytest.mark.parametrize("runtime", (False, True), ids=("static", "runtime"))
@pytest.mark.parametrize("prefix", (1, 5, None), ids=("one", "several", "full"))
def test_prefix(api, kind, runtime, prefix):
    width = _WIDTHS[kind]
    prefix = width if prefix is None else prefix

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int64):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS // width))
        if cutlass.const_expr(runtime):
            result = api.reduce(
                _group(api, kind),
                inputs[thread],
                binary_op="max",
                broadcast=False,
                valid_items=count,
            )
        else:
            result = api.reduce(
                _group(api, kind),
                inputs[thread],
                binary_op="max",
                broadcast=False,
                valid_items=prefix,
            )
        if thread % width == 0:
            outputs[thread // width] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, count: cutlass.Int64):
        kernel(source, observed, count).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS, shift=23)
    observed = np.zeros(_THREADS // width, dtype=np.int32)
    expected = np.array(
        [row[:prefix].max() for row in source.reshape(-1, width)], dtype=np.int32
    )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, prefix)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("algorithm", _ALGORITHMS)
@pytest.mark.parametrize("items", (0, 2), ids=("scalar", "payload"))
def test_block_algorithm(api, algorithm, items):
    extent = max(items, 1)

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS * extent))
        outputs = cute.make_tensor(observed, cute.make_layout(1))
        checks = cute.make_tensor(preserved, cute.make_layout(_THREADS * extent))
        if cutlass.const_expr(items):
            value = api.ThreadData(items, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(items):
                value[item] = inputs[thread * items + item]
        else:
            value = inputs[thread]
        result = api.sum(api.this_block(), value, broadcast=False, algorithm=algorithm)
        if thread == 0:
            outputs[0] = result
        if cutlass.const_expr(items):
            for item in cutlass.range_constexpr(items):
                checks[thread * items + item] = value[item]
        else:
            checks[thread] = value

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        kernel(source, observed, preserved).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS * extent, shift=29)
    observed = np.zeros(1, dtype=np.int32)
    preserved = np.zeros_like(source)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, out, check)
    assert observed[0] == source.sum(dtype=np.int32)
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
def test_nonmembers(api):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, continued: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(_THREADS))
        checks = cute.make_tensor(continued, cute.make_layout(_THREADS))
        group = api.this_block().group_by(3, exhaustive=False)
        result = api.sum(group, inputs[thread])
        if thread < 96:
            outputs[thread] = result
        checks[thread] = inputs[thread] + 1

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, continued: cute.Pointer):
        kernel(source, observed, continued).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _THREADS, shift=37)
    observed = np.full_like(source, -101)
    continued = np.zeros_like(source)
    expected = observed.copy()
    expected[:96] = source[:96].sum(dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(observed) as out,
        device_array(continued) as check,
    ):
        launch(src, out, check)
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(continued, source + 1)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
def test_cluster(api):
    cutlass.cuda.initialize_cuda_context()
    device = check_cuda(driver.cuCtxGetDevice())
    supported = check_cuda(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH, device
        )
    )
    if not supported:
        pytest.skip("device does not support thread-block cluster launch")
    block_threads = 32
    cluster_threads = 64
    total_threads = 128
    fields = 8

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        x, y, _ = cute.arch.thread_idx()
        block_index = cute.arch.block_idx()[0]
        thread = x + 8 * y
        index = block_index * block_threads + thread
        inputs = cute.make_tensor(source, cute.make_layout(total_threads))
        outputs = cute.make_tensor(observed, cute.make_layout(fields * total_threads))
        cluster = api.this_cluster()
        cluster.sync()
        cluster.sync_aligned()
        outputs[index] = api.sum(cluster, inputs[index])
        payload = api.ThreadData(2, dtype=cutlass.Int32)
        payload[0] = inputs[index]
        payload[1] = inputs[index] + 3
        outputs[total_threads + index] = api.reduce(cluster, payload, binary_op="max")
        outputs[2 * total_threads + index] = cluster.rank()
        outputs[3 * total_threads + index] = cluster.count()
        outputs[4 * total_threads + index] = api.this_block().rank("cluster")
        outputs[5 * total_threads + index] = cluster.rank("grid")
        outputs[6 * total_threads + index] = api.this_grid().count("cluster")
        outputs[7 * total_threads + index] = api.this_grid().count("block")

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(
            grid=(4, 1, 1), block=(8, 4, 1), cluster=(2, 1, 1)
        )

    source = values_for(np.int32, total_threads, shift=71)
    observed = np.zeros(fields * total_threads, dtype=np.int64)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out)
    indices = np.arange(total_threads)
    expected = np.stack(
        (
            np.repeat(source.reshape(-1, cluster_threads).sum(axis=1), cluster_threads),
            np.repeat(
                source.reshape(-1, cluster_threads).max(axis=1) + 3, cluster_threads
            ),
            indices % cluster_threads,
            np.full(total_threads, cluster_threads),
            (indices // block_threads) % 2,
            indices // cluster_threads,
            np.full(total_threads, 2),
            np.full(total_threads, 4),
        )
    )
    np.testing.assert_array_equal(observed.reshape(fields, total_threads), expected)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("kind", ("block", "warp", "logical"))
def test_reuse_loop(api, kind):
    width = _WIDTHS[kind]
    groups = _THREADS // width
    count = width - 3

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(groups))
        total = cutlass.Int32(0)
        for iteration in range(iterations):
            result = api.sum(
                _group(api, kind),
                inputs[thread] + iteration,
                broadcast=False,
                valid_items=count,
            )
            if thread % width == 0:
                total += result
        if thread % width == 0:
            outputs[thread // width] = total

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        kernel(source, observed, iterations).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _THREADS, shift=43)
    observed = np.zeros(groups, dtype=np.int32)
    iterations = 8
    expected = np.array(
        [
            iterations * row[:count].sum(dtype=np.int32)
            + count * sum(range(iterations))
            for row in source.reshape(-1, width)
        ],
        dtype=np.int32,
    )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "kind,value",
    [
        (kind, value)
        for kind, width in (("block", _THREADS), ("warp", 32), ("logical", 8))
        for value in (-1, 0, width + 1, (1 << 32) + 1)
    ],
)
def test_bad_prefix(tmp_path, kind, value):
    path = tmp_path / "invalid_prefix.py"
    path.write_text(f"""import numpy as np
import cutlass
from cutlass import cute
from cuda import coop
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array

assert cutlass_coop.__file__ == {cutlass_coop.__file__!r}

@cute.kernel
def kernel(source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int64):
    thread = cute.arch.thread_idx()[0]
    inputs = cute.make_tensor(source, cute.make_layout({_THREADS}))
    outputs = cute.make_tensor(destination, cute.make_layout(1))
    group = {"coop.this_block()" if kind == "block" else "coop.this_warp()" if kind == "warp" else "coop.this_warp().group_by(8)"}
    result = coop.sum(group, inputs[thread], broadcast=False, valid_items=count)
    if thread == 0:
        outputs[0] = result

@cute.jit
def launch(source: cute.Pointer, destination: cute.Pointer, count: cutlass.Int64):
    kernel(source, destination, count).launch(grid=1, block={_THREADS})

with device_array(np.ones({_THREADS}, dtype=np.int32)) as src, device_array(np.zeros(1, dtype=np.int32)) as dst:
    launch(src, dst, {value})
    status = driver.cuCtxSynchronize()[0]
    print(f"prefix launch status: {{int(status)}}", flush=True)
raise AssertionError("invalid Reduce prefix did not trap")
""")
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=180
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


@pytest.mark.parametrize("route", ("cudax", "cub"))
def test_final_cubin(tmp_path, route):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(_THREADS))
        outputs = cute.make_tensor(observed, cute.make_layout(1))
        if cutlass.const_expr(route == "cub"):
            result = cutlass_coop.sum(
                cutlass_coop.this_block(),
                inputs[thread],
                broadcast=False,
                algorithm="raking",
            )
        else:
            result = cutlass_coop.sum(
                cutlass_coop.this_block(), inputs[thread], broadcast=False
            )
        if thread == 0:
            outputs[0] = result

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _THREADS, shift=47)
    observed = np.zeros(1, dtype=np.int32)
    with device_array(source) as src, device_array(observed) as out:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, out)
        compiled(src, out)
    assert observed[0] == source.sum(dtype=np.int32)
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [cuobjdump, "--dump-sass", str(cubin)], text=True
        )
        cubin.with_suffix(".sass").write_text(sass)
        assert "cuda_coop_cutlass_" not in sass
        assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/reduce.py"
    spec = importlib.util.spec_from_file_location("cutlass_reduce_example", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)
