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


# --- Known numba-cuda-mlir upstream failures -------------------------------
#
# These tests fail because of bugs/limitations in numba-cuda-mlir (not in
# cuda.compute).  Each is xfail'd against the tracking issue.  strict=False
# because some are data-dependent (e.g. a comparison bug only shows for certain
# value ranges) and may pass; an xpass simply flags that the issue is resolved.
# Remove a rule once its upstream issue is fixed.
_UNSIGNED_DTYPES = ("uint8", "uint16", "uint32", "uint64")


def _upstream_xfail_reason(name: str, nodeid: str):
    """Return an xfail reason for a known numba-cuda-mlir failure, else None.

    ``name`` is the test function name (without parametrization); ``nodeid``
    carries the parametrization, used where only some parameter sets fail.
    """

    def issue(num, text):
        return f"numba-cuda-mlir#{num}: {text}"

    # E (#123): the ** operator lowers to mismatched-type ops (cmpi / powf).
    # The reduce-over-transform-iterator case squares an integer (cmpi); the
    # transform-output-iterator cases square a float, where only float32 hits
    # the powf type mismatch (float64 lowers cleanly).
    if name == "test_transform_iterator":
        return issue(123, "`**` operator lowers to mismatched-type ops")
    if (
        name
        in (
            "test_reduce_transform_output_iterator",
            "test_segmented_reduce_transform_output_iterator",
        )
        and "float32" in nodeid
    ):
        return issue(123, "`**` operator lowers to mismatched-type ops")

    # G (#124): no device array-from-pointer, so captured-array state cannot be
    # used with array ops (cuda.atomic, len, .shape).
    if name in (
        "test_unary_transform_stateful_counting",
        "test_select_stateful_atomic",
        "test_select_with_side_effect_counting_rejects",
        "test_stateful_transform_same_bytecode_different_sizes",
    ):
        return issue(124, "no device array-from-pointer for captured-array state")

    # D (#121): integer comparisons ignore operand signedness.
    if name == "test_select_reuse_object" and any(
        f"[{d}]" in nodeid for d in ("uint64", "int8", "int16", "int32")
    ):
        return issue(121, "integer comparison ignores signedness")
    if (
        name.startswith("test_merge_sort")
        and "compare_op" in nodeid
        and any(d in nodeid for d in _UNSIGNED_DTYPES)
    ):
        return issue(121, "unsigned integer comparison compiled as signed")

    # C (#120): a complex value loaded through a CPointer fails to lower.
    if name in (
        "test_complex_device_reduce",
        "test_unique_by_key_complex",
        "test_merge_sort_keys_complex",
    ):
        return issue(120, "complex value loaded through a CPointer fails to lower")
    if (
        name
        in (
            "test_scan_array_input",
            "test_segmented_reduce",
            "test_unary_transform",
            "test_binary_transform",
        )
        and "complex" in nodeid
    ):
        return issue(120, "complex value loaded through a CPointer fails to lower")

    # A (#119): "__numba_cuda_mlir_error_code" symbol multiply defined when an
    # algorithm links more than one operator.  (same_predicate links a single
    # deduplicated op and is fine.)
    if (
        "test_three_way_partition.py" in nodeid
        and name != "test_three_way_partition_same_predicate"
    ):
        return issue(119, "duplicate __numba_cuda_mlir_error_code on multi-op link")
    if (
        name
        in (
            "test_device_sum_map_mul2_count_it",
            "test_device_sum_map_mul2_cp_array_it",
            "test_device_sum_map_mul_map_mul_count_it",
        )
        and "[False-" in nodeid
    ):
        return issue(119, "duplicate __numba_cuda_mlir_error_code on multi-op link")
    if name in (
        "test_reducer_caching",
        "test_reduce_struct_type_minmax",
        "test_device_segmented_reduce_for_rowwise_sum",
        "test_zip_iterator_with_counting_iterator_and_transform",
    ):
        return issue(119, "duplicate __numba_cuda_mlir_error_code on multi-op link")

    return None


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.get_closest_marker("no_numba"):
            if "raise_on_numba_import" not in item.fixturenames:
                item.fixturenames.append("raise_on_numba_import")

        name = getattr(item, "originalname", None) or item.name.split("[")[0]
        reason = _upstream_xfail_reason(name, item.nodeid)
        if reason is not None:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
