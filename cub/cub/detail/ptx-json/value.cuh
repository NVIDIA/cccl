// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/detail/ptx-json/string.cuh>

#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

namespace ptx_json
{
template <auto V>
struct value;

template <typename T>
struct is_value : cuda::std::false_type
{};

template <auto V>
struct is_value<value<V>> : cuda::std::true_type
{};

template <typename T>
concept a_value = is_value<T>::value;

template <a_value auto Nested>
struct value<Nested>
{
  __device__ consteval static auto emit()
  {
    value<Nested>::emit();
  }
};

__device__ constexpr int len10(int V)
{
  int count = 0;
  do
  {
    V /= 10;
    ++count;
  } while (V != 0);
  return count;
}

template <cuda::std::integral auto V>
struct value<V>
{
  __device__ consteval static auto emit()
  {
    static_assert(V >= 0, "Only non-negative integers are supported");
    constexpr auto l10   = len10(V);
    char buffer[l10 + 1] = {};
    auto buffer_ptr      = buffer + l10;
    auto init            = V;
    do
    {
      *--buffer_ptr = '0' + init % 10;
      init /= 10;
    } while (init != 0);
    return string(buffer);
  }
};

template <>
struct value<true>
{
  __device__ consteval static auto emit()
  {
    return string("true");
  }
};

template <>
struct value<false>
{
  __device__ consteval static auto emit()
  {
    return string("false");
  }
};

#pragma nv_diag_suppress 842
template <int N, string<N> V>
struct value<V>
{
#pragma nv_diag_default 842
  __device__ consteval static auto emit()
  {
    return string("\"", V, "\"");
  }
};
}; // namespace ptx_json
