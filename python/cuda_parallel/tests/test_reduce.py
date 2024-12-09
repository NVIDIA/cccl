# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import cupy as cp
import numpy
import pytest
import random
import numba.cuda
import numba.types
import cuda.parallel.experimental as cudax
from cuda.parallel.experimental import _iterators
from cuda.parallel.experimental import iterators


def random_int(shape, dtype):
    return numpy.random.randint(0, 5, size=shape).astype(dtype)


def type_to_problem_sizes(dtype):
    if dtype in [numpy.uint8, numpy.int8]:
        return [2, 4, 5, 6]
    elif dtype in [numpy.uint16, numpy.int16]:
        return [4, 8, 12, 14]
    elif dtype in [numpy.uint32, numpy.int32]:
        return [16, 20, 24, 28]
    elif dtype in [numpy.uint64, numpy.int64]:
        return [16, 20, 24, 28]
    else:
        raise ValueError("Unsupported dtype")


@pytest.mark.parametrize(
    "dtype", [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]
)
def test_device_reduce(dtype):
    def op(a, b):
        return a + b

    init_value = 42
    h_init = numpy.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items_pow2 in type_to_problem_sizes(dtype):
        num_items = 2**num_items_pow2
        h_input = random_int(num_items, dtype)
        d_input = numba.cuda.to_device(h_input)
        temp_storage_size = reduce_into(None, d_input, d_output, None, h_init)
        d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=numpy.uint8)
        reduce_into(d_temp_storage, d_input, d_output, None, h_init)
        h_output = d_output.copy_to_host()
        assert h_output[0] == sum(h_input) + init_value


def test_complex_device_reduce():
    def op(a, b):
        return a + b

    h_init = numpy.array([40.0 + 2.0j], dtype=complex)
    d_output = numba.cuda.device_array(1, dtype=complex)
    reduce_into = cudax.reduce_into(d_output, d_output, op, h_init)

    for num_items in [42, 420000]:
        h_input = numpy.random.random(num_items) + 1j * numpy.random.random(num_items)
        d_input = numba.cuda.to_device(h_input)
        temp_storage_bytes = reduce_into(None, d_input, d_output, None, h_init)
        d_temp_storage = numba.cuda.device_array(temp_storage_bytes, numpy.uint8)
        reduce_into(d_temp_storage, d_input, d_output, None, h_init)

        result = d_output.copy_to_host()[0]
        expected = numpy.sum(h_input, initial=h_init[0])
        assert result == pytest.approx(expected)


def test_device_reduce_dtype_mismatch():
    def min_op(a, b):
        return a if a < b else b

    dtypes = [numpy.int32, numpy.int64]
    h_inits = [numpy.array([], dt) for dt in dtypes]
    h_inputs = [numpy.array([], dt) for dt in dtypes]
    d_outputs = [numba.cuda.device_array(1, dt) for dt in dtypes]
    d_inputs = [numba.cuda.to_device(h_inp) for h_inp in h_inputs]

    reduce_into = cudax.reduce_into(d_inputs[0], d_outputs[0], min_op, h_inits[0])

    for ix in range(3):
        with pytest.raises(
            TypeError, match=r"^dtype mismatch: __init__=int32, __call__=int64$"
        ):
            reduce_into(
                None,
                d_inputs[int(ix == 0)],
                d_outputs[int(ix == 1)],
                None,
                h_inits[int(ix == 2)],
            )


def _test_device_sum_with_iterator(
    l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
):
    def add_op(a, b):
        return a + b

    expected_result = start_sum_with
    for v in l_varr:
        expected_result = add_op(expected_result, v)

    if use_numpy_array:
        h_input = numpy.array(l_varr, dtype_inp)
        d_input = numba.cuda.to_device(h_input)
    else:
        d_input = i_input

    d_output = numba.cuda.device_array(1, dtype_out)  # to store device sum

    h_init = numpy.array([start_sum_with], dtype_out)

    reduce_into = cudax.reduce_into(
        d_in=d_input, d_out=d_output, op=add_op, init=h_init
    )

    temp_storage_size = reduce_into(
        None, d_in=d_input, d_out=d_output, num_items=len(l_varr), init=h_init
    )
    d_temp_storage = numba.cuda.device_array(temp_storage_size, dtype=numpy.uint8)

    reduce_into(d_temp_storage, d_input, d_output, len(l_varr), h_init)

    h_output = d_output.copy_to_host()
    assert h_output[0] == expected_result


def mul2(val):
    return 2 * val


def mul3(val):
    return 3 * val


SUPPORTED_VALUE_TYPE_NAMES = (
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
)


@pytest.fixture(params=SUPPORTED_VALUE_TYPE_NAMES)
def supported_value_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy_array(request):
    return request.param


@pytest.mark.parametrize(
    "type_obj_from_str", [_iterators.numba_type_from_any, numpy.dtype, cp.dtype]
)
def test_value_type_name_round_trip(type_obj_from_str, supported_value_type):
    # If all round trip tests here pass for all value types we are supporting,
    # this provides a super easy way to support numba.types, numpy.dtypes,
    # cupy.dtypes and plain strings as `value_type` arguments.
    type_obj = type_obj_from_str(supported_value_type)
    assert str(type_obj) == supported_value_type


def test_device_sum_raw_pointer_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    # Exercise non-public _iterators.pointer() independently from iterators.TransformIterator().
    rng = random.Random(0)
    l_varr = [rng.randrange(100) for _ in range(num_items)]
    dtype_inp = numpy.dtype(supported_value_type)
    dtype_out = dtype_inp
    raw_pointer_devarr = numba.cuda.to_device(numpy.array(l_varr, dtype=dtype_inp))
    i_input = _iterators.pointer(
        raw_pointer_devarr, _iterators.numba_type_from_any(supported_value_type)
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_cache_modified_input_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    rng = random.Random(0)
    l_varr = [rng.randrange(100) for _ in range(num_items)]
    dtype_inp = numpy.dtype(supported_value_type)
    dtype_out = dtype_inp
    input_devarr = numba.cuda.to_device(numpy.array(l_varr, dtype=dtype_inp))
    i_input = iterators.CacheModifiedInputIterator(
        input_devarr, value_type=supported_value_type, modifier="stream"
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_constant_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [42 for distance in range(num_items)]
    dtype_inp = numpy.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = iterators.ConstantIterator(42, value_type=supported_value_type)
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_counting_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [start_sum_with + distance for distance in range(num_items)]
    dtype_inp = numpy.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = iterators.CountingIterator(
        start_sum_with, value_type=supported_value_type
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    "value_type_name_pair",
    list(zip(SUPPORTED_VALUE_TYPE_NAMES, SUPPORTED_VALUE_TYPE_NAMES))
    + [
        ("float32", "int16"),
        ("float32", "int32"),
        ("float64", "int32"),
        ("float64", "int64"),
        ("int64", "float32"),
    ],
)
def test_device_sum_map_mul2_count_it(
    use_numpy_array, value_type_name_pair, num_items=3, start_sum_with=10
):
    l_varr = [2 * (start_sum_with + distance) for distance in range(num_items)]
    vtn_out, vtn_inp = value_type_name_pair
    dtype_inp = numpy.dtype(vtn_inp)
    dtype_out = numpy.dtype(vtn_out)
    i_input = iterators.TransformIterator(
        mul2,
        iterators.CountingIterator(start_sum_with, value_type=vtn_inp),
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    ("fac_out", "fac_mid", "vtn_out", "vtn_mid", "vtn_inp"),
    [
        (3, 2, "int32", "int32", "int32"),
        (2, 2, "float64", "float32", "int16"),
    ],
)
def test_device_sum_map_mul_map_mul_count_it(
    use_numpy_array,
    fac_out,
    fac_mid,
    vtn_out,
    vtn_mid,
    vtn_inp,
    num_items=3,
    start_sum_with=10,
):
    l_varr = [
        fac_out * (fac_mid * (start_sum_with + distance))
        for distance in range(num_items)
    ]
    dtype_inp = numpy.dtype(vtn_inp)
    dtype_out = numpy.dtype(vtn_out)
    mul_funcs = {2: mul2, 3: mul3}
    i_input = iterators.TransformIterator(
        mul_funcs[fac_out],
        iterators.TransformIterator(
            mul_funcs[fac_mid],
            iterators.CountingIterator(start_sum_with, value_type=vtn_inp),
        ),
    )
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    "value_type_name_pair",
    [
        ("int32", "int32"),
        ("int64", "int32"),
        ("int32", "int64"),
    ],
)
def test_device_sum_map_mul2_cp_array_it(
    use_numpy_array, value_type_name_pair, num_items=3, start_sum_with=10
):
    vtn_out, vtn_inp = value_type_name_pair
    dtype_inp = numpy.dtype(vtn_inp)
    dtype_out = numpy.dtype(vtn_out)
    rng = random.Random(0)
    l_d_in = [rng.randrange(100) for _ in range(num_items)]
    a_d_in = cp.array(l_d_in, dtype_inp)
    i_input = iterators.TransformIterator(mul2, a_d_in)
    l_varr = [mul2(v) for v in l_d_in]
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )
