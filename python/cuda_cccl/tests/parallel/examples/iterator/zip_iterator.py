# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Zip iterator example demonstrating reduction with zip iterator.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def zip_iterator_elementwise_sum_example():
    """Demonstrate elementwise sum with zip iterator combining two arrays."""

    # Create input arrays
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input2 = cp.array([10, 20, 30, 40, 50], dtype=np.int32)

    # Create zip iterator that pairs elements from both arrays
    zip_it = parallel.ZipIterator(d_input1, d_input2)

    # Create output array to store the elementwise sums
    num_items = len(d_input1)
    d_output = cp.empty(num_items, dtype=np.int32)

    # Define the transform operation that extracts and sums the paired values
    def sum_paired_values(pair):
        """Extract values from the zip iterator pair and sum them."""
        # The zip iterator provides tuples, so we access elements by index
        return pair[0] + pair[1]

    # Apply the unary transform to perform elementwise sum
    parallel.unary_transform(zip_it, d_output, sum_paired_values, num_items)

    # Calculate expected results
    expected = d_input1.get() + d_input2.get()  # [11, 22, 33, 44, 55]
    result = d_output.get()

    # Verify results
    np.testing.assert_allclose(result, expected)

    print(f"Input array 1: {d_input1.get()}")
    print(f"Input array 2: {d_input2.get()}")
    print(f"Elementwise sum result: {result}")
    print(f"Expected result: {expected}")

    return result


def zip_iterator_reduction_example():
    """Demonstrate reduction with zip iterator combining two arrays."""

    @parallel.gpu_struct
    class Pair:
        first: np.int32
        second: np.float32

    def sum_pairs(p1, p2):
        """Reduction operation that adds corresponding elements of pairs."""
        return Pair(p1[0] + p2[0], p1[1] + p2[1])

    # Create input arrays
    d_input1 = cp.array([1, 2, 3, 4, 5], dtype=np.int32)
    d_input2 = cp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    # Create zip iterator that pairs elements from both arrays
    zip_it = parallel.ZipIterator(d_input1, d_input2)

    # Set up reduction
    h_init = Pair(0, 0.0)  # Initial value for the reduction
    d_output = cp.empty(1, dtype=Pair.dtype)  # Storage for output

    # Run reduction
    parallel.reduce_into(zip_it, d_output, sum_pairs, len(d_input1), h_init)

    # Calculate expected results
    expected_first = sum(d_input1.get())  # 1+2+3+4+5 = 15
    expected_second = sum(d_input2.get())  # 1.0+2.0+3.0+4.0+5.0 = 15.0

    result = d_output.get()[0]
    assert result["first"] == expected_first
    assert result["second"] == expected_second

    print(
        f"Zip iterator result: first={result['first']} (expected: {expected_first}), "
        f"second={result['second']} (expected: {expected_second})"
    )
    return result


def zip_iterator_counting_reduction_example():
    """Demonstrate zip iterator with counting iterator and array."""

    @parallel.gpu_struct
    class IndexValuePair:
        index: np.int32
        value: np.int32

    def max_by_value(p1, p2):
        """Reduction operation that returns the pair with the larger value."""
        return p1 if p1[1] > p2[1] else p2

    # Create a counting iterator for indices and an array for values
    counting_it = parallel.CountingIterator(np.int32(0))  # 0, 1, 2, 3, 4, 5, 6, 7
    arr = cp.asarray([0, 1, 2, 4, 7, 3, 5, 6], dtype=np.int32)

    # Zip iterator pairs indices with values
    zip_it = parallel.ZipIterator(counting_it, arr)

    num_items = 8
    h_init = IndexValuePair(-1, -1)  # Initial value
    d_output = cp.empty(1, dtype=IndexValuePair.dtype)

    # Run reduction to find the index with maximum value
    parallel.reduce_into(zip_it, d_output, max_by_value, num_items, h_init)

    result = d_output.get()[0]
    expected_index = 4  # Index of maximum value (7) in the array
    expected_value = 7

    assert result["index"] == expected_index
    assert result["value"] == expected_value

    print(
        f"Zip iterator with counting result: index={result['index']} "
        f"(expected: {expected_index}), value={result['value']} (expected: {expected_value})"
    )
    return result


if __name__ == "__main__":
    print("Running zip iterator examples...")
    zip_iterator_elementwise_sum_example()
    zip_iterator_reduction_example()
    zip_iterator_counting_reduction_example()
    print("Zip iterator examples completed successfully!")
