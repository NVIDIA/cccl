// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/detail/ptx-json/object.cuh>

namespace ptx_json
{
template <typename T>
concept not_a_keyed_value = !is_keyed_value<T>::value;

template <auto... KV>
struct array;

template <>
struct array<>
{
  __forceinline__ __device__ static void emit()
  {
    asm volatile("[]");
  }
};

template <not_a_keyed_value auto First, not_a_keyed_value auto... NKVs>
struct array<First, NKVs...>
{
  __device__ consteval static auto emit()
  {
    return string("[", value<First>::emit(), string(comma(), value<NKVs>::emit())..., "]");
  }
};

template <typename T>
struct is_array : cuda::std::false_type
{};

template <auto... NKV>
struct is_array<object<NKV...>> : cuda::std::true_type
{};
} // namespace ptx_json
