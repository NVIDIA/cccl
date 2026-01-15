// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/detail/ptx-json/string.cuh>
#include <cub/detail/ptx-json/value.cuh>

#include <cuda/std/__type_traits/integral_constant.h>

namespace ptx_json
{
template <auto K, typename V>
struct keyed_value
{
  __device__ consteval static auto emit()
  {
    return string(value<K>::emit(), ":", V::emit());
  }
};

template <typename T>
struct is_keyed_value : cuda::std::false_type
{};

template <auto K, typename V>
struct is_keyed_value<keyed_value<K, V>> : cuda::std::true_type
{};

template <typename T>
concept a_keyed_value = is_keyed_value<T>::value;

template <auto... KV>
struct object;

template <>
struct object<>
{
  __device__ consteval static auto emit()
  {
    return string("{}");
  }
};

template <a_keyed_value auto First, a_keyed_value auto... KVs>
struct object<First, KVs...>
{
  __device__ consteval static auto emit()
  {
    return string("{", First.emit(), string(comma(), KVs.emit())..., "}");
  }
};

template <typename T>
struct is_object : cuda::std::false_type
{};

template <auto... KV>
struct is_object<object<KV...>> : cuda::std::true_type
{};

template <string V>
struct key
{
  template <typename U>
  __device__ consteval keyed_value<V, U> operator=(U u)
  {
    return {};
  }
};
} // namespace ptx_json
