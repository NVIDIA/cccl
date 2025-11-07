// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

/**
 * Custom comparator that simply uses `operator <` of the given type.
 */
struct custom_less_op_t
{
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs)
  {
    return lhs < rhs;
  }
};

/**
 * Custom comparator that compares a tuple type's first element using `operator <`.
 */
struct compare_first_lt_op_t
{
  /**
   * We need to be able to have two different types for lhs and rhs, as the call to std::stable_sort with a
   * zip-iterator, will pass a thrust::tuple for lhs and a tuple_of_iterator_references for rhs.
   */
  template <typename LhsT, typename RhsT>
  __host__ __device__ bool operator()(const LhsT& lhs, const RhsT& rhs) const
  {
    return thrust::get<0>(lhs) < thrust::get<0>(rhs);
  }
};

/**
 * Function object to computes the modulo of a given value. Used within sort tests to reduce the value-range of sort
 * keys and, hence, cause more ties between sort keys.
 */
template <typename T>
struct mod_op_t
{
  T mod;
  __host__ __device__ T operator()(T val) const
  {
    return val % mod;
  }
};
