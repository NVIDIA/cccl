# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Histogram examples demonstrating the histogram_even algorithm.
"""

import cupy as cp
import numpy as np

import cuda.cccl.parallel.experimental as parallel


def basic_histogram_example():
    """Demonstrate basic histogram operation with floating-point data."""

    # Input samples covering different bins
    h_samples = np.array(
        [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 11.8], dtype="float32"
    )
    d_samples = cp.asarray(h_samples)

    # Configure histogram with 6 bins from 0 to 12
    num_levels = 7  # 6 bins = 7 levels
    h_num_output_levels = np.array([num_levels], dtype="int32")
    h_lower_level = np.array([0.0], dtype="float64")
    h_upper_level = np.array([12.0], dtype="float64")

    d_histogram = cp.zeros(num_levels - 1, dtype="int32")

    # Run histogram with automatic temp storage allocation
    parallel.histogram_even(
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_samples),
    )

    # Check the result
    h_result = cp.asnumpy(d_histogram)

    # Expected bins:
    # Bin 0 [0-2): 1 sample (0.3)
    # Bin 1 [2-4): 5 samples (2.2, 2.9, 3.5, 2.9, 2.1)
    # Bin 2 [4-6): 0 samples
    # Bin 3 [6-8): 3 samples (6.1, 7.1, 6.1)
    # Bin 4 [8-10): 0 samples
    # Bin 5 [10-12): 1 sample (11.8)
    expected_histogram = np.array([1, 5, 0, 3, 0, 1], dtype="int32")

    assert np.array_equal(h_result, expected_histogram)
    print(f"Input samples: {h_samples}")
    print(f"Histogram result: {h_result}")
    print("Bin ranges: [0-2), [2-4), [4-6), [6-8), [8-10), [10-12)")
    print("Basic histogram example passed.")
    return h_result


def image_histogram_example():
    """Demonstrate histogram for image-like data (simulated grayscale values)."""

    # Simulate grayscale image pixel values (0-255)
    np.random.seed(42)  # For reproducible results
    h_pixels = np.random.randint(0, 256, size=1000, dtype="uint8").astype("float32")
    d_pixels = cp.asarray(h_pixels)

    # Create 16 bins for grayscale values
    num_levels = 17  # 16 bins = 17 levels
    h_num_output_levels = np.array([num_levels], dtype="int32")
    h_lower_level = np.array([0.0], dtype="float64")
    h_upper_level = np.array([256.0], dtype="float64")

    d_histogram = cp.zeros(num_levels - 1, dtype="int32")

    # Run histogram with automatic temp storage allocation
    parallel.histogram_even(
        d_pixels,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_pixels),
    )

    # Check the result
    h_result = cp.asnumpy(d_histogram)

    # Verify all pixels are counted
    total_count = np.sum(h_result)
    assert total_count == len(h_pixels)

    print("Image histogram (grayscale simulation):")
    print(f"Total pixels: {len(h_pixels)}")
    print(f"Histogram bins (16 bins): {h_result}")
    print(f"Total counted: {total_count}")
    print(f"Min pixel value: {np.min(h_pixels)}")
    print(f"Max pixel value: {np.max(h_pixels)}")
    print("Image histogram example passed.")
    return h_result


def integer_data_histogram_example():
    """Demonstrate histogram with integer data types."""

    # Integer data with known distribution
    h_data = np.array(
        [1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 7, 8, 9, 9], dtype="int32"
    ).astype("float32")
    d_data = cp.asarray(h_data)

    # Create 10 bins from 0 to 10
    num_levels = 11  # 10 bins = 11 levels
    h_num_output_levels = np.array([num_levels], dtype="int32")
    h_lower_level = np.array([0.0], dtype="float64")
    h_upper_level = np.array([10.0], dtype="float64")

    d_histogram = cp.zeros(num_levels - 1, dtype="int32")

    # Run histogram with automatic temp storage allocation
    parallel.histogram_even(
        d_data,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        len(h_data),
    )

    # Check the result
    h_result = cp.asnumpy(d_histogram)

    # Expected distribution:
    # Bin 0 [0-1): 0 samples
    # Bin 1 [1-2): 2 samples (1, 1)
    # Bin 2 [2-3): 3 samples (2, 2, 2)
    # Bin 3 [3-4): 1 sample (3)
    # Bin 4 [4-5): 4 samples (4, 4, 4, 4)
    # Bin 5 [5-6): 2 samples (5, 5)
    # Bin 6 [6-7): 1 sample (6)
    # Bin 7 [7-8): 1 sample (7)
    # Bin 8 [8-9): 1 sample (8)
    # Bin 9 [9-10): 2 samples (9, 9)
    expected_histogram = np.array([0, 2, 3, 1, 4, 2, 1, 1, 1, 2], dtype="int32")

    assert np.array_equal(h_result, expected_histogram)
    print("Integer data histogram:")
    print(f"Input data: {h_data}")
    print(f"Histogram result: {h_result}")
    print(f"Total samples: {np.sum(h_result)}")
    print("Integer data histogram example passed.")
    return h_result


if __name__ == "__main__":
    print("Running histogram examples...")
    basic_histogram_example()
    print()
    image_histogram_example()
    print()
    integer_data_histogram_example()
    print("All histogram examples completed successfully!")
