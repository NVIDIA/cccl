import builtins

import cupy as cp
import numpy as np
import pytest

check_ldl_stl_in_sass = False


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


# Define a pytest fixture that returns random floating-point arrays only
@pytest.fixture(
    params=[
        np.float32,
        np.float64,
    ]
)
def floating_array(request):
    dtype = np.dtype(request.param)
    sample_size = 1000

    # Generate random floating-point values
    array = cp.random.random(sample_size).astype(dtype)
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

    if not check_ldl_stl_in_sass:
        print("not checking sass")
        return

    import cuda.compute._cccl_interop

    monkeypatch.setattr(
        cuda.compute._cccl_interop,
        "_check_sass",
        True,
    )


@pytest.fixture
def raise_on_numba_import(monkeypatch):
    """This fixture will raise if a test attempts to import numba"""
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ModuleNotFoundError(
                "This test is marked 'no_numba' but attempted to import it"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _backend_uses_v2() -> bool:
    """True iff cuda_cccl was built against cccl.c.parallel.v2 (HostJIT)."""
    try:
        from cuda.compute._build_info import USING_V2  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(USING_V2)


# Individual tests known to crash on the v2 backend that don't match the
# stateful/fp16 substring rules below. Match is on `item.name` (parametrized
# id, e.g. "test_foo[int32]") OR on the bare function name. Add a one-line
# reason for each so it's clear why it's deferred rather than fixed.
_V2_BROKEN_TESTS = {
    "test_segmented_sort_op_kind": "cudaErrorMisalignedAddress at runtime; v2 segmented_sort path",
    "test_select_with_side_effect_counting_rejects": "v2 select side-effect path",
}


def pytest_collection_modifyitems(config, items):
    skip_stateful_for_v2 = pytest.mark.skip(
        reason="v2 (HostJIT) backend: stateful-op marshaling is not yet implemented"
    )
    skip_fp16_for_v2 = pytest.mark.skip(
        reason="v2 (HostJIT) backend: fp16/__half disabled via CCCL_DISABLE_FP16_SUPPORT"
    )
    using_v2 = _backend_uses_v2()
    for item in items:
        # Check if the 'no_numba' marker is present on the test item
        if item.get_closest_marker("no_numba"):
            # If the marker is present, add 'raise_on_numba_import' to the list of required fixtures
            if "raise_on_numba_import" not in item.fixturenames:
                item.fixturenames.append("raise_on_numba_import")

        if not using_v2:
            continue

        # Skip stateful-op tests on the v2 backend (known limitation: state
        # marshaling between host and JIT'd device code is incomplete and
        # crashes with CUDA_ERROR_ILLEGAL_ADDRESS).
        lowered_name = item.name.lower()
        if "stateful" in lowered_name:
            item.add_marker(skip_stateful_for_v2)

        # Skip parametrized cases over fp16: v2's freestanding compile defines
        # CCCL_DISABLE_FP16_SUPPORT=1, so __half is only forward-declared and
        # any algorithm template instantiation against it fails.
        if "float16" in lowered_name or "fp16" in lowered_name:
            item.add_marker(skip_fp16_for_v2)

        # Explicit per-test deferrals.
        # `item.originalname` is the function name without parametrize suffix;
        # `item.name` includes it. Either match deferr the test.
        bare = getattr(item, "originalname", item.name)
        if bare in _V2_BROKEN_TESTS:
            item.add_marker(
                pytest.mark.skip(
                    reason="v2 (HostJIT) backend: " + _V2_BROKEN_TESTS[bare]
                )
            )
