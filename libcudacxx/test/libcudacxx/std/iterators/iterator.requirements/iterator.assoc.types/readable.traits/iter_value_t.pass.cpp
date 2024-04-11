//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class T>
// using iter_value_t;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#ifndef TEST_COMPILER_MSVC_2017 // MSVC 2017 cannot make this a constexpr function
template <class T, class Expected>
__host__ __device__ constexpr bool check_iter_value_t()
{
  constexpr bool result = cuda::std::same_as<cuda::std::iter_value_t<T>, Expected>;
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const&&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T volatile&&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_value_t<T const volatile&&>, Expected> == result);

  return result;
}

static_assert(check_iter_value_t<int*, int>());
static_assert(check_iter_value_t<int[], int>());
static_assert(check_iter_value_t<int[10], int>());

struct both_members
{
  using value_type   = double;
  using element_type = double;
};
static_assert(check_iter_value_t<both_members, double>());
#endif // !TEST_COMPILER_MSVC_2017

// clang-format off
template <class T, class = void>
inline constexpr bool check_no_iter_value_t = true;

template <class T>
inline constexpr bool check_no_iter_value_t<T, cuda::std::void_t<cuda::std::iter_value_t<T>>> = false;

static_assert(check_no_iter_value_t<void>);
static_assert(check_no_iter_value_t<double>);

struct S {};
static_assert(check_no_iter_value_t<S>);

struct different_value_element_members {
  using value_type = int;
  using element_type = long;
};
static_assert(check_no_iter_value_t<different_value_element_members>);

int main(int, char**) { return 0; }
