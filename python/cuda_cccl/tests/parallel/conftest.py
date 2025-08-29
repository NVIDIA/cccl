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
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
)
def input_array(request):
    dtype = np.dtype(request.param)
    sample_size = 1000

    # Generate random values based on the dtype
    if np.issubdtype(dtype, np.integer):
        is_unsigned = dtype.kind == "u"
        # For integer types, use np.random.randint for random integers
        if is_unsigned:
            low_inclusive, high_exclusive = 0, 8
        else:
            low_inclusive, high_exclusive = -5, 6
        array = cp.random.randint(
            low=low_inclusive, high=high_exclusive, size=sample_size, dtype=dtype
        )
    elif np.issubdtype(dtype, np.floating):
        # For floating-point types, use np.random.random and cast to the required dtype
        array = cp.random.random(sample_size).astype(dtype)
    elif np.issubdtype(dtype, np.complexfloating):
        # For complex types, generate random real and imaginary parts
        packed = cp.random.random(2 * sample_size)
        real_part = packed[:sample_size]
        imag_part = packed[sample_size:]
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


@pytest.fixture(scope="function", autouse=True)
def verify_sass(request, monkeypatch):
    if request.node.get_closest_marker("no_verify_sass"):
        return

    import cuda.cccl.parallel.experimental._cccl_interop

    monkeypatch.setattr(
        cuda.cccl.parallel.experimental._cccl_interop,
        "_check_sass",
        True,
    )
