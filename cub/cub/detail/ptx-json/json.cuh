// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/detail/ptx-json/array.cuh>
#include <cub/detail/ptx-json/object.cuh>
#include <cub/detail/ptx-json/string.cuh>
#include <cub/detail/ptx-json/value.cuh>

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstddef>

namespace ptx_json
{
template <auto V, typename = cuda::std::make_index_sequence<V.Length>>
const char reify[V.Length] = {};

template <int N, string<N> V, cuda::std::size_t... Is>
__device__ const char reify<V, cuda::std::index_sequence<Is...>>[] = {V.str[Is]...};

template <auto Tag>
struct tagged_json
{
  template <typename V, typename = cuda::std::enable_if_t<is_object<V>::value || is_array<V>::value>>
  __device__ consteval auto& operator=(V v)
  {
    return reify<string(
      "\ncccl.ptx_json.begin(", value<Tag>::emit(), ")", V::emit(), "cccl.ptx_json.end(", value<Tag>::emit(), ")\n")>;
  }
};

template <auto T>
__device__ consteval tagged_json<T> id()
{
  return {};
}
} // namespace ptx_json
