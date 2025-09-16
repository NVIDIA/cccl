// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/dispatch/dispatch_topk.cuh> // topk::select::{min, max}

#include <cuda/std/limits>

// Function object to generate monotonically non-decreasing values for small key types
template <typename T>
struct inc_t
{
  size_t num_item;
  double value_increment;

  // Needs to be default constructible to qualify as forward iterator
  inc_t() = default;

  inc_t(size_t num_item)
      : num_item(num_item)
  {
    if (num_item < cuda::std::numeric_limits<T>::max())
    {
      value_increment = 1;
    }
    else
    {
      value_increment = static_cast<double>(cuda::std::numeric_limits<T>::max()) / num_item;
    }
  }

  template <typename IndexT>
  __host__ __device__ T operator()(IndexT x)
  {
    return static_cast<T>(value_increment * x);
  }
};

template <cub::detail::topk::select SelectDirection>
using direction_to_comparator_t =
  cuda::std::conditional_t<SelectDirection == cub::detail::topk::select::min, cuda::std::less<>, cuda::std::greater<>>;
