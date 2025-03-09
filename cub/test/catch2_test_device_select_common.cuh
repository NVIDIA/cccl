// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_type.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

template <typename T>
struct less_than_t
{
  T compare;

  explicit __host__ less_than_t(T compare)
      : compare(compare)
  {}

  __host__ __device__ bool operator()(const T& a) const
  {
    return a < compare;
  }
};

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

template <typename SelectedItT, typename RejectedItT>
struct index_to_expected_partition_op
{
  using value_t = cub::detail::it_value_t<SelectedItT>;
  SelectedItT expected_selected_it;
  RejectedItT expected_rejected_it;
  std::int64_t expected_num_selected;

  template <typename OffsetT>
  __host__ __device__ value_t operator()(OffsetT index)
  {
    return (index < static_cast<OffsetT>(expected_num_selected))
           ? expected_selected_it[index]
           : expected_rejected_it[index - expected_num_selected];
  }
};

template <typename SelectedItT, typename RejectedItT>
static index_to_expected_partition_op<SelectedItT, RejectedItT> make_index_to_expected_partition_op(
  SelectedItT expected_selected_it, RejectedItT expected_rejected_it, std::int64_t expected_num_selected)
{
  return index_to_expected_partition_op<SelectedItT, RejectedItT>{
    expected_selected_it, expected_rejected_it, expected_num_selected};
}
