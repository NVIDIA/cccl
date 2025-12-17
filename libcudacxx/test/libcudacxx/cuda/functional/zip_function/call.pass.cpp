//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

struct foo
{};

struct Immutable
{
  constexpr Immutable() = default;

  __host__ __device__ constexpr int operator()(const int a, const double b, foo) const noexcept
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a, const double) const
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a) const noexcept
  {
    return a + 41;
  }
};

struct Mutable
{
  constexpr Mutable() = default;

  __host__ __device__ constexpr int operator()(const int a, const double b, foo)
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a, const double) noexcept
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a)
  {
    return a + 41;
  }
};

struct Mixed
{
  constexpr Mixed() = default;

  __host__ __device__ constexpr int operator()(const int a, const double b, foo) const noexcept
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a, const double) const
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a) const noexcept
  {
    return a + 41;
  }

  __host__ __device__ constexpr int operator()(const int a, const double b, foo)
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a, const double) noexcept
  {
    return a + 41;
  }
  __host__ __device__ constexpr int operator()(const int a)
  {
    return a + 41;
  }
};

template <bool IsNoexcept, class Fn, class Tuple>
__host__ __device__ constexpr void test(Fn&& fun, Tuple&& tuple)
{
  static_assert(cuda::std::is_invocable_v<Fn, Tuple>);
  static_assert(cuda::std::is_nothrow_invocable_v<Fn, Tuple> == IsNoexcept);
  assert(cuda::std::forward<Fn>(fun)(cuda::std::forward<Tuple>(tuple)) == 42);
}

__host__ __device__ constexpr bool test()
{
  using cuda::zip_function;

  cuda::std::tuple three_args{1, 3.14, foo{}};
  cuda::std::pair two_args{1, 3.14};
  cuda::std::tuple one_arg{1};
  {
    const zip_function<Immutable> fn{};
    test<true>(fn, three_args);
    test<false>(fn, two_args);
    test<true>(fn, one_arg);

    // Ensure we can also call the function with const arguments
    test<true>(fn, cuda::std::as_const(three_args));
    test<false>(fn, cuda::std::as_const(two_args));
    test<true>(fn, cuda::std::as_const(one_arg));

    // Ensure we can also call the function with prvalues
    test<true>(fn, cuda::std::tuple{1, 3.14, foo{}});
    test<false>(fn, cuda::std::pair{1, 3.14});
    test<true>(fn, cuda::std::tuple{1});
  }

  {
    zip_function<Mutable> fn{};
    test<false>(fn, three_args);
    test<true>(fn, two_args);
    test<false>(fn, one_arg);

    // Ensure we can also call the function with const arguments
    test<false>(fn, cuda::std::as_const(three_args));
    test<true>(fn, cuda::std::as_const(two_args));
    test<false>(fn, cuda::std::as_const(one_arg));

    // Ensure we can also call the function with prvalues
    test<false>(fn, cuda::std::tuple{1, 3.14, foo{}});
    test<true>(fn, cuda::std::pair{1, 3.14});
    test<false>(fn, cuda::std::tuple{1});
  }

  { // Ensure that we properly dispatch to the const overload then possible
    const zip_function<Mixed> const_fn{};
    test<true>(const_fn, three_args);
    test<false>(const_fn, two_args);
    test<true>(const_fn, one_arg);

    // Ensure we can also call the function with const arguments
    test<true>(const_fn, cuda::std::as_const(three_args));
    test<false>(const_fn, cuda::std::as_const(two_args));
    test<true>(const_fn, cuda::std::as_const(one_arg));

    // Ensure we can also call the function with prvalues
    test<true>(const_fn, cuda::std::tuple{1, 3.14, foo{}});
    test<false>(const_fn, cuda::std::pair{1, 3.14});
    test<true>(const_fn, cuda::std::tuple{1});

    zip_function<Mixed> fn{};
    test<false>(fn, three_args);
    test<true>(fn, two_args);
    test<false>(fn, one_arg);

    // Ensure we can also call the function with const arguments
    test<false>(fn, cuda::std::as_const(three_args));
    test<true>(fn, cuda::std::as_const(two_args));
    test<false>(fn, cuda::std::as_const(one_arg));

    // Ensure we can also call the function with prvalues
    test<false>(fn, cuda::std::tuple{1, 3.14, foo{}});
    test<true>(fn, cuda::std::pair{1, 3.14});
    test<false>(fn, cuda::std::tuple{1});
  }

  { // Ensure that we can instantiate a zip_function that is not invocable
    static_assert(!cuda::std::is_invocable_v<const zip_function<Mutable>, cuda::std::tuple<int, double, foo>>);
    static_assert(!cuda::std::is_invocable_v<const zip_function<Mutable>, cuda::std::pair<int, double>>);
    static_assert(!cuda::std::is_invocable_v<const zip_function<Mutable>, cuda::std::tuple<int>>);
  }

  return true;
};

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
