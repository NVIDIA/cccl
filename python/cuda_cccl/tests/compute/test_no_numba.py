import cupy as cp
import numpy as np
import pytest

import cuda.compute
from cuda.compute import OpKind

# Mainly, these tests check that we can use algorithms with OpKind
# operators while not requiring numba to be installed.
pytestmark = pytest.mark.no_numba


@pytest.mark.no_numba
def test_import_numba_raises():
    with pytest.raises(
        ImportError, match="This test is marked 'no_numba' but attempted to import it"
    ):
        import numba.cuda  # noqa: F401


def test_reduce_op_kind():
    num_items = 100
    h_input = np.arange(num_items, dtype=np.int32)
    d_input = cp.array(h_input)
    d_output = cp.empty(1, dtype=np.int32)

    h_init = np.array(0, dtype=np.int32)
    cuda.compute.reduce_into(d_input, d_output, OpKind.PLUS, num_items, h_init)

    result = d_output.get()[0]
    expected = np.sum(h_input)
    assert result == expected


def test_binary_transform_op_kind():
    num_items = 100
    h_input1 = np.arange(num_items, dtype=np.int32)
    h_input2 = np.arange(num_items, dtype=np.int32) * 2
    d_input1 = cp.array(h_input1)
    d_input2 = cp.array(h_input2)
    d_output = cp.empty(num_items, dtype=np.int32)

    cuda.compute.binary_transform(d_input1, d_input2, d_output, OpKind.PLUS, num_items)

    result = d_output.get()
    expected = h_input1 + h_input2
    assert np.array_equal(result, expected)


def test_segmented_sort_op_kind():
    # Create segments: [3, 1, 4] | [1, 5, 9, 2] | [6, 5]
    num_items = 9
    h_keys = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5], dtype=np.int32)
    h_offsets = np.array([0, 3, 7, 9], dtype=np.int32)

    d_keys_in = cp.array(h_keys)
    d_keys_out = cp.empty(num_items, dtype=np.int32)
    d_offsets = cp.array(h_offsets)

    num_segments = len(h_offsets) - 1

    cuda.compute.segmented_sort(
        d_keys_in,
        d_keys_out,
        None,
        None,
        num_items,
        num_segments,
        d_offsets[:-1],
        d_offsets[1:],
        cuda.compute.SortOrder.ASCENDING,
    )

    result = d_keys_out.get()
    # Expected: [1, 3, 4] | [1, 2, 5, 9] | [5, 6]
    expected = np.array([1, 3, 4, 1, 2, 5, 9, 5, 6], dtype=np.int32)
    assert np.array_equal(result, expected)
