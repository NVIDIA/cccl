/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/detail/ptx-json/array.h>
#include <cub/detail/ptx-json/object.h>
#include <cub/detail/ptx-json/string.h>
#include <cub/detail/ptx-json/value.h>

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
