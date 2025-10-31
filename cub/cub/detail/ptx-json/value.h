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

#include <cub/detail/ptx-json/string.h>

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
