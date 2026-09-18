# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Array and scalar Shuffle oracles, controls, and scratch reuse."""

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
_ITEMS = 3
_TILE = _THREADS * _ITEMS


def _run_array(api, dtype, mode, *, block=_BLOCK):
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
        payload = api.ThreadData(_ITEMS, dtype=value_type, alignment=64)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        result = api.shuffle(api.this_block(), payload, mode=mode)
        assert result.alignment >= 64
        if cutlass.const_expr(mode == "down"):
            if thread == threads - 1:
                result[_ITEMS - 1] = value_type(0)
        else:
            if thread == 0:
                result[0] = value_type(0)
        assert result.dtype is value_type
        assert result.alignment >= 64
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = result[item]
            checks[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, preserved: cute.Pointer):
        kernel(source, observed, preserved).launch(grid=1, block=block)

    source = values_for(dtype, size, shift=47)
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    expected = np.zeros_like(source)
    if mode == "down":
        expected[:-1] = source[1:]
    else:
        expected[1:] = source[:-1]
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
@pytest.mark.parametrize("mode", ("up", "down"))
def test_array_types(api, dtype, mode):
    _run_array(api, dtype, mode)


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("mode", ("up", "down"))
def test_partial_block(api, mode):
    _run_array(api, np.int64, mode, block=(8, 3, 2))


def _run_scalar(dtype, mode, *, runtime, distance=5, distance_dtype=np.int64):
    value_type = cutlass_dtype(dtype)
    control_type = cutlass_dtype(distance_dtype)

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        controls: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
    ):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(_THREADS)), value_type
        )
        distances = cute.recast_tensor(
            cute.make_tensor(controls, cute.make_layout(_THREADS)), control_type
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(observed, cute.make_layout(_THREADS)), value_type
        )
        checks = cute.recast_tensor(
            cute.make_tensor(preserved, cute.make_layout(_THREADS)), value_type
        )
        value = inputs[thread]
        if cutlass.const_expr(runtime):
            result = cutlass_coop.shuffle(
                cutlass_coop.this_block(), value, mode=mode, distance=distances[thread]
            )
        else:
            result = cutlass_coop.shuffle(
                cutlass_coop.this_block(), value, mode=mode, distance=distance
            )
        assert isinstance(result, value_type)
        outputs[thread] = result
        checks[thread] = value

    @cute.jit
    def launch(
        source: cute.Pointer,
        controls: cute.Pointer,
        observed: cute.Pointer,
        preserved: cute.Pointer,
    ):
        kernel(source, controls, observed, preserved).launch(grid=1, block=_BLOCK)

    source = values_for(dtype, _THREADS, shift=53)
    if runtime:
        distances = np.arange(_THREADS) % 9
        if mode == "rotate":
            distances += 1
            distances[-1] = _THREADS - 1
        elif np.dtype(distance_dtype).kind == "i":
            distances -= 4
        distances = distances.astype(distance_dtype)
    else:
        distances = np.full(_THREADS, distance, dtype=distance_dtype)
    observed = np.zeros_like(source)
    preserved = np.zeros_like(source)
    selected = np.arange(_THREADS) + distances.astype(np.int64)
    if mode == "rotate":
        selected %= _THREADS
    defined = (selected >= 0) & (selected < _THREADS)
    with (
        device_array(source) as src,
        device_array(distances) as control,
        device_array(observed) as out,
        device_array(preserved) as check,
    ):
        launch(src, control, out, check)
    np.testing.assert_array_equal(observed[defined], source[selected[defined]])
    np.testing.assert_array_equal(preserved, source)


@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
@pytest.mark.parametrize("mode", ("offset", "rotate"))
@pytest.mark.parametrize("runtime", (False, True), ids=("static", "runtime"))
def test_scalar_types(dtype, mode, runtime):
    _run_scalar(dtype, mode, runtime=runtime)


@pytest.mark.parametrize(
    "distance_dtype",
    (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64),
)
@pytest.mark.parametrize("mode", ("offset", "rotate"))
def test_control_types(distance_dtype, mode):
    _run_scalar(np.int32, mode, runtime=True, distance_dtype=distance_dtype)


@pytest.mark.parametrize(
    "mode,distance",
    (
        ("offset", -64),
        ("offset", -1),
        ("offset", 0),
        ("offset", 1),
        ("offset", 64),
        ("rotate", 1),
        ("rotate", 63),
    ),
)
def test_static_edges(mode, distance):
    _run_scalar(np.int32, mode, runtime=False, distance=distance)


@pytest.mark.parametrize(
    "mode,distance",
    (
        ("rotate", 0),
        ("rotate", -1),
        ("rotate", 64),
        ("rotate", (1 << 32) + 1),
        ("offset", -(1 << 31) - 1),
        ("offset", 1 << 31),
        ("offset", (1 << 32) + 1),
    ),
)
def test_bad_distance(tmp_path, mode, distance):
    path = tmp_path / "invalid_distance.py"
    path.write_text(f"""import numpy as np
import cutlass
from cutlass import cute
from cuda.bindings import driver
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array

assert cutlass_coop.__file__ == {cutlass_coop.__file__!r}

@cute.kernel
def kernel(source: cute.Pointer, observed: cute.Pointer, distance: cutlass.Int64):
    thread = cute.arch.thread_idx()[0]
    inputs = cute.make_tensor(source, cute.make_layout(64))
    outputs = cute.make_tensor(observed, cute.make_layout(64))
    outputs[thread] = cutlass_coop.shuffle(cutlass_coop.this_block(), inputs[thread], mode={mode!r}, distance=distance)

@cute.jit
def launch(source: cute.Pointer, observed: cute.Pointer, distance: cutlass.Int64):
    kernel(source, observed, distance).launch(grid=1, block=64)

with device_array(np.ones(64, dtype=np.int32)) as src, device_array(np.zeros(64, dtype=np.int32)) as out:
    launch(src, out, {distance})
    status = driver.cuCtxSynchronize()[0]
    print(f"shuffle launch status: {{int(status)}}", flush=True)
raise AssertionError("invalid Shuffle distance did not trap")
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
        f"shuffle launch status: {int(status)}" in output
        for status in (
            driver.CUresult.CUDA_ERROR_ILLEGAL_INSTRUCTION,
            driver.CUresult.CUDA_ERROR_LAUNCH_FAILED,
        )
    ), output


@pytest.mark.parametrize("api", _APIS, ids=("common", "qualified"))
@pytest.mark.parametrize("mixed", (False, True), ids=("shuffle", "exchange-shuffle"))
def test_reuse_loop(api, mixed):
    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        x, y, z = cute.arch.thread_idx()
        thread = x + _BLOCK[0] * (y + _BLOCK[1] * z)
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        payload = api.ThreadData(_ITEMS, dtype=cutlass.Int32)
        for item in cutlass.range_constexpr(_ITEMS):
            payload[item] = inputs[thread * _ITEMS + item]
        for iteration in range(iterations):
            if cutlass.const_expr(mixed):
                payload = api.exchange(
                    api.this_block(), payload, mode="blocked_to_striped"
                )
            payload = api.shuffle(api.this_block(), payload, mode="down")
            if thread == _THREADS - 1:
                payload[_ITEMS - 1] = cutlass.Int32(0)
        for item in cutlass.range_constexpr(_ITEMS):
            outputs[thread * _ITEMS + item] = payload[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32):
        kernel(source, observed, iterations).launch(grid=1, block=_BLOCK)

    source = values_for(np.int32, _TILE, shift=61)
    observed = np.zeros_like(source)
    expected = source.copy()
    iterations = 5
    for _ in range(iterations):
        if mixed:
            expected = expected.reshape(_ITEMS, _THREADS).T.reshape(-1)
        expected = np.concatenate((expected[1:], np.zeros(1, dtype=np.int32)))
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("scalar", (False, True), ids=("array-down", "scalar-rotate"))
def test_final_cubin(tmp_path, scalar):
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        pytest.skip("cuobjdump is required to inspect final linked instructions")

    @cute.kernel
    def kernel(source: cute.Pointer, observed: cute.Pointer):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(_TILE))
        outputs = cute.make_tensor(observed, cute.make_layout(_TILE))
        if cutlass.const_expr(scalar):
            outputs[thread] = cutlass_coop.shuffle(
                cutlass_coop.this_block(), inputs[thread], mode="rotate", distance=7
            )
        else:
            payload = cutlass_coop.ThreadData(_ITEMS, dtype=cutlass.Int32)
            for item in cutlass.range_constexpr(_ITEMS):
                payload[item] = inputs[thread * _ITEMS + item]
            result = cutlass_coop.shuffle(
                cutlass_coop.this_block(), payload, mode="down"
            )
            if thread == _THREADS - 1:
                result[_ITEMS - 1] = cutlass.Int32(0)
            for item in cutlass.range_constexpr(_ITEMS):
                outputs[thread * _ITEMS + item] = result[item]

    @cute.jit
    def launch(source: cute.Pointer, observed: cute.Pointer):
        kernel(source, observed).launch(grid=1, block=_THREADS)

    source = values_for(np.int32, _TILE, shift=67)
    observed = np.zeros_like(source)
    with device_array(source) as src, device_array(observed) as out:
        compiled = cute.compile[(KeepCUBIN, DumpDir(str(tmp_path)))](launch, src, out)
        compiled(src, out)
    if scalar:
        np.testing.assert_array_equal(
            observed[:_THREADS], np.roll(source[:_THREADS], -7)
        )
    else:
        np.testing.assert_array_equal(
            observed, np.concatenate((source[1:], np.zeros(1, dtype=np.int32)))
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


@pytest.mark.parametrize("api", ("common", "qualified"))
def test_example(api):
    path = PACKAGE_ROOT / "examples/cutlass/exchange_shuffle.py"
    spec = importlib.util.spec_from_file_location(
        "cutlass_exchange_shuffle_example", path
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    example.run_example(api)
