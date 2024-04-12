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
// using iter_difference_t;

#include <cuda/std/concepts>
#include <cuda/std/iterator>

#if TEST_STD_VER > 2017
template <class T>
inline constexpr bool has_no_iter_difference_t = !requires { typename cuda::std::iter_difference_t<T>; };

#else
template <class T, class = void>
inline constexpr bool has_no_iter_difference_t = true;

template <class T>
inline constexpr bool has_no_iter_difference_t<T, cuda::std::void_t<cuda::std::iter_difference_t<T>>> = false;
#endif

#ifndef TEST_COMPILER_MSVC_2017 // MSVC 2017 cannot make this a constexpr function
template <class T, class Expected>
__host__ __device__ constexpr bool check_iter_difference_t()
{
  constexpr bool result = cuda::std::same_as<cuda::std::iter_difference_t<T>, Expected>;
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T volatile>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const volatile>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T volatile&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const volatile&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const&&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T volatile&&>, Expected> == result);
  static_assert(cuda::std::same_as<cuda::std::iter_difference_t<T const volatile&&>, Expected> == result);

  return result;
}

static_assert(check_iter_difference_t<int, int>());
static_assert(check_iter_difference_t<int*, cuda::std::ptrdiff_t>());

struct int_subtraction
{
  __host__ __device__ friend int operator-(int_subtraction, int_subtraction);
};
static_assert(check_iter_difference_t<int_subtraction, int>());
#endif // !TEST_COMPILER_MSVC_2017

static_assert(has_no_iter_difference_t<void>);
static_assert(has_no_iter_difference_t<double>);

struct S
{};
static_assert(has_no_iter_difference_t<S>);

struct void_subtraction
{
  __host__ __device__ friend void operator-(void_subtraction, void_subtraction);
};
static_assert(has_no_iter_difference_t<void_subtraction>);

int main(int, char**)
{
  return 0;
}
