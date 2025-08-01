# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Histogram example demonstrating the object API.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def histogram_object_example():
    """Demonstrate histogram using the object API with make_histogram_even."""

    # Input samples with known distribution
    h_samples = np.array(
        [1.5, 2.3, 4.7, 6.2, 7.8, 3.1, 5.5, 8.9, 2.7, 6.4], dtype="float32"
    )
    d_samples = cp.asarray(h_samples)

    # Configure histogram with 5 bins from 0 to 10
    num_levels = 6  # 5 bins = 6 levels
    h_num_output_levels = np.array([num_levels], dtype="int32")
    h_lower_level = np.array([0.0], dtype="float64")
    h_upper_level = np.array([10.0], dtype="float64")

    d_histogram = cp.zeros(num_levels - 1, dtype="int32")

    # Create histogram object
    histogrammer = parallel.make_histogram_even(
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_samples),
    )

    # First call to get temp storage size
    temp_storage_size = histogrammer(
        None,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_samples),
    )

    # Allocate temp storage and run histogram
    d_temp_storage = cp.empty(temp_storage_size, dtype=np.uint8)
    histogrammer(
        d_temp_storage,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_samples),
    )

    # Check the result
    h_result = cp.asnumpy(d_histogram)

    # Expected distribution:
    # Bin 0 [0-2): 1 sample (1.5)
    # Bin 1 [2-4): 3 samples (2.3, 3.1, 2.7)
    # Bin 2 [4-6): 2 samples (4.7, 5.5)
    # Bin 3 [6-8): 3 samples (6.2, 7.8, 6.4)
    # Bin 4 [8-10): 1 sample (8.9)
    expected_histogram = np.array([1, 3, 2, 3, 1], dtype="int32")

    print(f"Input samples: {h_samples}")
    print(f"Histogram object API result: {h_result}")
    print("Bin ranges: [0-2), [2-4), [4-6), [6-8), [8-10)")
    print(f"Temp storage size: {temp_storage_size} bytes")

    np.testing.assert_array_equal(h_result, expected_histogram)
    print("Histogram object API example passed.")


if __name__ == "__main__":
    print("Running histogram_object_example...")
    histogram_object_example()
    print("All examples completed successfully!")
