// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/transform.h>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/device_policy.h>
#include <c2h/vector.h>

struct CustomLess
{
  template <typename T>
  [[nodiscard]] __device__ __host__ bool operator()(const T& lhs, const T& rhs) const
  {
    return lhs < rhs;
  }

  template <typename T>
  [[nodiscard]] static __device__ __host__ T get_oob_default()
  {
    if constexpr (cuda::std::is_same_v<T, float>)
    {
      return cuda::std::numeric_limits<T>::infinity();
    }
    else
    {
      return cuda::std::numeric_limits<T>::max();
    }
  }
};

// Maps keys that are not ordered strictly before the upper bound to KeyT{}.
template <typename KeyT>
struct clamp_key_below_t
{
  KeyT upper_bound;

  __device__ __host__ KeyT operator()(const KeyT& key) const
  {
    return CustomLess{}(key, upper_bound) ? key : KeyT{};
  }
};

// Partial Sort() and TopK() overloads with an OOB sentinel require every valid key ordered strictly before the
// sentinel, which c2h::gen does not guarantee since it can generate the sentinel value itself.
// This clamping method works generically, also for custom_t
template <typename KeyT>
void clamp_keys_below(c2h::device_vector<KeyT>& d_keys, KeyT upper_bound)
{
  thrust::transform(
    c2h::device_policy, d_keys.begin(), d_keys.end(), d_keys.begin(), clamp_key_below_t<KeyT>{upper_bound});
}
