import cupy as cp
import numpy as np
import pytest


# Define a pytest fixture that returns random arrays with different dtypes
@pytest.fixture(
    params=[
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
        np.complex128,
    ]
)
def input_array(request):
    dtype = request.param

    # Generate random values based on the dtype
    if np.issubdtype(dtype, np.integer):
        # For integer types, use np.random.randint for random integers
        array = cp.random.randint(low=0, high=10, size=1000, dtype=dtype)
    elif np.issubdtype(dtype, np.floating):
        # For floating-point types, use np.random.random and cast to the required dtype
        array = cp.random.random(1000).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # For complex types, generate random real and imaginary parts
        real_part = cp.random.random(1000)
        imag_part = cp.random.random(1000)
        array = (real_part + 1j * imag_part).astype(dtype)

    return array


class Stream:
    """
    Simple cupy stream wrapper that implements the __cuda_stream__ protocol.
    """

    def __init__(self, cp_stream):
        self.cp_stream = cp_stream

    def __cuda_stream__(self):
        return (0, self.cp_stream.ptr)

    @property
    def ptr(self):
        return self.cp_stream.ptr


@pytest.fixture(scope="function")
def cuda_stream() -> Stream:
    return Stream(cp.cuda.Stream())
