// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/limits>

template <typename T>
struct inc_t
{
  size_t num_item;
  double value_increment;

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
  __host__ __device__ T operator()(IndexT x) const
  {
    return static_cast<T>(value_increment * x);
  }
};
