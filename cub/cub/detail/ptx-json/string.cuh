// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
