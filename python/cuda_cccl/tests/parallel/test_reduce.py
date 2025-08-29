# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import random

import cupy as cp
import numba.cuda
import numba.types
import numpy as np
import pytest

import cuda.cccl.parallel.experimental as parallel


def random_int(shape, dtype):
    return np.random.randint(0, 5, size=shape).astype(dtype)


def type_to_problem_sizes(dtype):
    if dtype in [np.uint8, np.int8]:
        return [2, 4, 5, 6]
    elif dtype in [np.float16]:
        return [4, 8, 10]
    elif dtype in [np.uint16, np.int16]:
        return [4, 8, 12, 14]
    elif dtype in [np.uint32, np.int32]:
        return [16, 20, 24, 26]
    elif dtype in [np.uint64, np.int64]:
        return [16, 20, 24, 25]
    else:
        raise ValueError("Unsupported dtype")


def get_mark(dt, log_size):
    if log_size + np.log2(np.dtype(dt).itemsize) < 21:
        return tuple()
    return pytest.mark.large


def add_op(a, b):
    return a + b


reduce_params = [
    pytest.param(
        dt,
        2**log_size,
        parallel.OpKind.PLUS if dt == np.float16 else add_op,
        marks=get_mark(dt, log_size),
    )
    for dt in [np.uint8, np.uint16, np.uint32, np.uint64, np.float16]
    for log_size in type_to_problem_sizes(dt)
]


@pytest.mark.parametrize("dtype,num_items,op", reduce_params)
def test_device_reduce(dtype, num_items, op):
    init_value = 42
    h_init = np.array([init_value], dtype=dtype)
    d_output = numba.cuda.device_array(1, dtype=dtype)

    h_input = random_int(num_items, dtype)
    d_input = numba.cuda.to_device(h_input)

    parallel.reduce_into(d_input, d_output, op, d_input.size, h_init)
    h_output = d_output.copy_to_host()
    assert h_output[0] == pytest.approx(
        sum(h_input) + init_value, rel=0.08 if dtype == np.float16 else 0
    )  # obtained relative error value from c2h/include/c2h/check_results.cuh


def test_complex_device_reduce():
    h_init = np.array([40.0 + 2.0j], dtype=complex)
    d_output = numba.cuda.device_array(1, dtype=complex)

    for num_items in [42, 420000]:
        real_imag = np.random.random((2, num_items))
        h_input = real_imag[0] + 1j * real_imag[1]
        d_input = numba.cuda.to_device(h_input)
        assert d_input.size == num_items
        parallel.reduce_into(d_input, d_output, add_op, num_items, h_init)

        result = d_output.copy_to_host()[0]
        expected = np.sum(h_input, initial=h_init[0])
        assert result == pytest.approx(expected)


def _test_device_sum_with_iterator(
    l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
):
    expected_result = start_sum_with
    for v in l_varr:
        expected_result = add_op(expected_result, v)

    if use_numpy_array:
        h_input = np.array(l_varr, dtype_inp)
        d_input = numba.cuda.to_device(h_input)
    else:
        d_input = i_input

    d_output = numba.cuda.device_array(1, dtype_out)  # to store device sum

    h_init = np.array([start_sum_with], dtype_out)

    parallel.reduce_into(d_input, d_output, add_op, len(l_varr), h_init)

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
    # "float16", # TODO: this doesn't work with iterators right now due to a numba ctypes error
    "float32",
    "float64",
)


@pytest.fixture(params=SUPPORTED_VALUE_TYPE_NAMES)
def supported_value_type(request):
    return request.param


@pytest.fixture(params=[True, False])
def use_numpy_array(request):
    return request.param


def test_device_sum_cache_modified_input_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    rng = random.Random(0)
    l_varr = [rng.randrange(100) for _ in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    input_devarr = numba.cuda.to_device(np.array(l_varr, dtype=dtype_inp))
    i_input = parallel.CacheModifiedInputIterator(input_devarr, modifier="stream")
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_constant_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [42 for distance in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = parallel.ConstantIterator(dtype_inp.type(42))
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_device_sum_counting_it(
    use_numpy_array, supported_value_type, num_items=3, start_sum_with=10
):
    l_varr = [start_sum_with + distance for distance in range(num_items)]
    dtype_inp = np.dtype(supported_value_type)
    dtype_out = dtype_inp
    i_input = parallel.CountingIterator(dtype_inp.type(start_sum_with))
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


@pytest.mark.parametrize(
    "value_type_name_pair",
    list(zip(SUPPORTED_VALUE_TYPE_NAMES, SUPPORTED_VALUE_TYPE_NAMES))
    + [
        # ("float16", "int16"), # TODO: this doesn't work with numba right now due to an unresolved extern operator definition
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
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    i_input = parallel.TransformIterator(
        parallel.CountingIterator(dtype_inp.type(start_sum_with)), mul2
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
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    mul_funcs = {2: mul2, 3: mul3}
    i_input = parallel.TransformIterator(
        parallel.TransformIterator(
            parallel.CountingIterator(dtype_inp.type(start_sum_with)),
            mul_funcs[fac_mid],
        ),
        mul_funcs[fac_out],
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
    dtype_inp = np.dtype(vtn_inp)
    dtype_out = np.dtype(vtn_out)
    rng = random.Random(0)
    l_d_in = [rng.randrange(100) for _ in range(num_items)]
    a_d_in = cp.array(l_d_in, dtype_inp)
    i_input = parallel.TransformIterator(a_d_in, mul2)
    l_varr = [mul2(v) for v in l_d_in]
    _test_device_sum_with_iterator(
        l_varr, start_sum_with, i_input, dtype_inp, dtype_out, use_numpy_array
    )


def test_reducer_caching():
    def sum_op(x, y):
        return x + y

    # inputs are device arrays
    reducer_1 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are device arrays of different dtype:
    reducer_1 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int32"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # outputs are of different dtype:
    reducer_1 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int32"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # inputs are of same dtype but different size
    # (should still use cached reducer):
    reducer_1 = parallel.make_reduce_into(
        cp.zeros(3, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        cp.zeros(5, dtype="int64"),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are counting iterators of the
    # same value type:
    reducer_1 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are counting iterators of different value type:
    reducer_1 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int64(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    def op1(x):
        return x

    def op2(x):
        return 2 * x

    def op3(x):
        return x

    # inputs are TransformIterators
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators with different
    # op:
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op2),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2

    # inputs are TransformIterators with same op
    # but different name:
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op3),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )

    # inputs are CountingIterators of same kind
    # but different state:
    reducer_1 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int32(0)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.CountingIterator(np.int32(1)),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )

    assert reducer_1 is reducer_2

    # inputs are TransformIterators of same kind
    # but different state:
    ary1 = cp.asarray([0, 1, 2], dtype="int64")
    ary2 = cp.asarray([0, 1], dtype="int64")
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(ary1, op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(ary2, op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators of same kind
    # but different state:
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(1)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is reducer_2

    # inputs are TransformIterators with different kind:
    reducer_1 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int32(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    reducer_2 = parallel.make_reduce_into(
        parallel.TransformIterator(parallel.CountingIterator(np.int64(0)), op1),
        cp.zeros(1, dtype="int64"),
        sum_op,
        np.zeros(1, dtype="int64"),
    )
    assert reducer_1 is not reducer_2


@pytest.fixture(params=[True, False])
def array_2d(request):
    f_contiguous = request.param
    arr = cp.random.rand(5, 10)
    if f_contiguous:
        try:
            return cp.asfortranarray(arr)
        except ImportError:  # cublas unavailable
            return arr
    else:
        return arr


def test_reduce_2d_array(array_2d):
    def binary_op(x, y):
        return x + y

    d_out = cp.empty(1, dtype=array_2d.dtype)
    h_init = np.asarray([0], dtype=array_2d.dtype)
    d_in = array_2d
    parallel.reduce_into(d_in, d_out, binary_op, d_in.size, h_init)
    np.testing.assert_allclose(d_in.sum().get(), d_out.get())


def test_reduce_non_contiguous():
    def binary_op(x, y):
        return x + y

    size = 10
    d_out = cp.empty(1, dtype="int64")
    h_init = np.asarray([0], dtype="int64")

    d_in = cp.zeros((size, 2))[:, 0]
    with pytest.raises(ValueError, match="Non-contiguous arrays are not supported."):
        _ = parallel.make_reduce_into(d_in, d_out, binary_op, h_init)

    d_in = cp.zeros(size)[::2]
    with pytest.raises(ValueError, match="Non-contiguous arrays are not supported."):
        _ = parallel.make_reduce_into(d_in, d_out, binary_op, h_init)


def test_reduce_with_stream(cuda_stream):
    def add_op(x, y):
        return x + y

    h_init = np.asarray([0], dtype=np.int32)
    h_in = random_int(5, np.int32)

    cp_stream = cp.cuda.ExternalStream(cuda_stream.ptr)
    with cp_stream:
        d_in = cp.asarray(h_in)
        d_out = cp.empty(1, dtype=np.int32)

    parallel.reduce_into(d_in, d_out, add_op, d_in.size, h_init, stream=cuda_stream)
    with cp_stream:
        cp.testing.assert_allclose(d_in.sum().get(), d_out.get())


def test_reduce_invalid_stream():
    # Invalid stream that doesn't implement __cuda_stream__
    class Stream1:
        def __init__(self):
            pass

    # Invalid stream that implements __cuda_stream__ but returns the wrong type
    class Stream2:
        def __init__(self):
            pass

        def __cuda_stream__(self):
            return None

    # Invalid stream that returns an invalid handle
    class Stream3:
        def __init__(self):
            pass

        def __cuda_stream__(self):
            return (0, None)

    def add_op(x, y):
        return x + y

    d_out = cp.empty(1)
    h_init = np.empty(1)
    d_in = cp.empty(1)
    reduce_into = parallel.make_reduce_into(d_in, d_out, add_op, h_init)

    with pytest.raises(
        TypeError, match="does not implement the '__cuda_stream__' protocol"
    ):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream1(),
        )

    with pytest.raises(
        TypeError, match="could not obtain __cuda_stream__ protocol version and handle"
    ):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream2(),
        )

    with pytest.raises(TypeError, match="invalid stream handle"):
        _ = reduce_into(
            None,
            d_in=d_in,
            d_out=d_out,
            num_items=d_in.size,
            h_init=h_init,
            stream=Stream3(),
        )


def test_device_reduce_well_known_plus():
    """Test reduce with well-known PLUS operation."""
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    dtype = np.int32
    h_init = np.array([0], dtype=dtype)
    d_input = cp.array([1, 2, 3, 4, 5], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known PLUS operation
    parallel.reduce_into(d_input, d_output, parallel.OpKind.PLUS, len(d_input), h_init)

    # Check the result is correct
    expected_output = 15  # 1+2+3+4+5
    assert (d_output == expected_output).all()


@pytest.mark.xfail(reason="MINIMUM op is not implemented. See GH #5515")
def test_device_reduce_well_known_minimum():
    """Test reduce with well-known MINIMUM operation."""
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    dtype = np.int32
    h_init = np.array([100], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MINIMUM operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MINIMUM, len(d_input), h_init
    )

    # Check the result is correct
    expected_output = 0  # minimum value
    assert (d_output == expected_output).all()


@pytest.mark.xfail(reason="MAXIMUM op is not implemented. See GH #5515")
def test_device_reduce_well_known_maximum():
    """Test reduce with well-known MAXIMUM operation."""
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    dtype = np.int32
    h_init = np.array([-100], dtype=dtype)
    d_input = cp.array([8, 6, 7, 5, 3, 0, 9], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MAXIMUM operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MAXIMUM, len(d_input), h_init
    )

    # Check the result is correct
    expected_output = 9  # maximum value
    assert (d_output == expected_output).all()


def test_device_reduce_well_known_multiplies():
    """Test reduce with well-known MULTIPLIES operation."""
    import cupy as cp
    import numpy as np

    import cuda.cccl.parallel.experimental as parallel

    dtype = np.int32
    h_init = np.array([1], dtype=dtype)
    d_input = cp.array([2, 3, 4], dtype=dtype)
    d_output = cp.empty(1, dtype=dtype)

    # Run reduction with well-known MULTIPLIES operation
    parallel.reduce_into(
        d_input, d_output, parallel.OpKind.MULTIPLIES, len(d_input), h_init
    )

    # Check the result is correct
    expected_output = 24  # 1*2*3*4
    assert (d_output == expected_output).all()
