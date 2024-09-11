// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

template <typename T>
struct mod_n
{
  T mod;
  __host__ __device__ bool operator()(T x)
  {
    return (x % mod == 0) ? true : false;
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  __host__ __device__ T operator()(T x)
  {
    return x * multiplier;
  }
};

template <typename T, typename TargetT>
struct modx_and_add_divy
{
  T mod;
  T div;

  __host__ __device__ TargetT operator()(T x)
  {
    return static_cast<TargetT>((x % mod) + (x / div));
  }
};