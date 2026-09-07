# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# example-begin
import cupy as cp

from cuda.compute.algorithms import select
from cuda.compute.iterators import TransformIterator


# Create iterator that squares each value
def square(x):
    return x * x


# Select squared values that are greater than 20
def greater_than_20(x):
    return x > 20


def main():
    # Create input data
    d_in = cp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=cp.int32)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.zeros(2, dtype=cp.uint64)

    squared_iter = TransformIterator(d_in, square)

    # Execute select
    select(
        d_in=squared_iter,
        d_out=d_out,
        d_num_selected_out=d_num_selected,
        cond=greater_than_20,
        num_items=len(d_in),
    )

    # Get results
    num_selected = int(d_num_selected[0])
    result = d_out[:num_selected].get()
    print(f"Selected {num_selected} items: {result}")
    # Output: Selected 4 items: [25 36 49 64]
    # (5^2=25, 6^2=36, 7^2=49, 8^2=64, all > 20)

    assert num_selected == 4
    assert (result == [25, 36, 49, 64]).all()


if __name__ == "__main__":
    main()
# example-end
