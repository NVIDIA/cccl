//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <string>

#include "literal.h"

enum class MyChar : char
{
};

template <>
struct std::char_traits<MyChar>
{};

template <>
struct cuda::std::char_traits<MyChar>
{
  using base = cuda::std::char_traits<char>;

  using char_type = MyChar;
  using int_type  = typename base::int_type;

  __host__ __device__ static constexpr void assign(char_type& __lhs, const char_type& __rhs) noexcept
  {
    __lhs = __rhs;
  }

  [[nodiscard]] __host__ __device__ static constexpr bool eq(char_type __lhs, char_type __rhs) noexcept
  {
    return __lhs == __rhs;
  }

  [[nodiscard]] __host__ __device__ static constexpr bool lt(char_type __lhs, char_type __rhs) noexcept
  {
    return static_cast<unsigned char>(__lhs) < static_cast<unsigned char>(__rhs);
  }

  [[nodiscard]] __host__ __device__ static constexpr int
  compare(const char_type* __lhs, const char_type* __rhs, size_t __count) noexcept
  {
    return ::cuda::std::__cccl_memcmp(__lhs, __rhs, __count);
  }

  [[nodiscard]] __host__ __device__ inline static size_t constexpr length(const char_type* __s) noexcept
  {
    return ::cuda::std::__cccl_strlen(__s);
  }

  [[nodiscard]] __host__ __device__ static constexpr const char_type*
  find(const char_type* __s, size_t __n, const char_type& __a) noexcept
  {
    return ::cuda::std::__cccl_memchr<const char_type>(__s, __a, __n);
  }

  __host__ __device__ static constexpr char_type* move(char_type* __s1, const char_type* __s2, size_t __n) noexcept
  {
    return ::cuda::std::__cccl_memmove(__s1, __s2, __n);
  }

  __host__ __device__ static constexpr char_type* copy(char_type* __s1, const char_type* __s2, size_t __n) noexcept
  {
    return ::cuda::std::__cccl_memcpy(__s1, __s2, __n);
  }

  __host__ __device__ static constexpr char_type* assign(char_type* __s, size_t __n, char_type __a) noexcept
  {
    return ::cuda::std::__cccl_memset(__s, __a, __n);
  }

  [[nodiscard]] __host__ __device__ static constexpr char_type to_char_type(int_type __c) noexcept
  {
    return char_type(__c);
  }

  [[nodiscard]] __host__ __device__ static constexpr int_type to_int_type(char_type __c) noexcept
  {
    return int_type(static_cast<unsigned char>(__c));
  }

  [[nodiscard]] __host__ __device__ static constexpr bool eq_int_type(int_type __lhs, int_type __rhs) noexcept
  {
    return __lhs == __rhs;
  }
};

bool test()
{
  [[maybe_unused]] cuda::std::basic_string_view<MyChar> str{reinterpret_cast<const MyChar*>("test")};
  return true;
}

int main(int, char**)
{
  return 0;
}
