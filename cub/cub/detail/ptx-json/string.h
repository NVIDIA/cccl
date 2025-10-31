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

#include <cuda/std/__type_traits/remove_cvref.h>

namespace ptx_json
{
#pragma nv_diag_suppress 177
template <int N>
struct string;

template <typename T>
struct string_part_len_impl;

template <int N>
struct string_part_len_impl<string<N>>
{
  static const constexpr auto value = N - 1;
};

template <int N>
struct string_part_len_impl<char[N]>
{
  static const constexpr auto value = N - 1;
};

template <typename T>
constexpr int string_part_len = string_part_len_impl<T>::value;

template <int N>
__device__ constexpr auto& get_string_part(const string<N>& str)
{
  return str.str;
}

template <int N>
__device__ constexpr auto& get_string_part(const char (&str)[N])
{
  return str;
}

template <int N>
struct string
{
  static const constexpr auto Length = N;

  template <typename... Ts>
  __device__ constexpr string(Ts&&... strings)
  {
    static_assert(N == (0 + ... + string_part_len<cuda::std::remove_cvref_t<Ts>>) +1);
    int offset = 0;
    (init_part(get_string_part(strings), offset), ...);
    str[offset] = 0;
  }

  template <int M>
  __device__ constexpr void init_part(const char (&other)[M], int& offset)
  {
    for (int i = 0; i < M - 1; ++i)
    {
      str[offset + i] = other[i];
    }
    offset += M - 1;
  }

  char str[N];
};

template <typename... Ts>
string(Ts&&...) -> string<(0 + ... + string_part_len<cuda::std::remove_cvref_t<Ts>>) +1>;
#pragma nv_diag_default 177

__device__ consteval auto comma()
{
  return string(",");
}
} // namespace ptx_json
