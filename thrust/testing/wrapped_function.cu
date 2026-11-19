// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/detail/function.h>

#include <cuda/std/type_traits>

#include <catch2/catch_test_macros.hpp>

struct int_greater_than_zero
{
  _CCCL_HOST_DEVICE bool operator()(int x) const
  {
    return x > 0;
  }
};

struct void_unary
{
  int* acc;

  _CCCL_HOST_DEVICE void operator()(int x) const
  {
    *acc += x;
  }
};

struct int_minus
{
  _CCCL_HOST_DEVICE int operator()(int a, int b) const
  {
    return a - b;
  }
};

struct int_identity
{
  _CCCL_HOST_DEVICE int operator()(int x) const
  {
    return x;
  }
};

TEST_CASE("WrappedFunctionBoolPredicate", "[wrapped_function]")
{
  const thrust::detail::wrapped_function<int_greater_than_zero> pred{int_greater_than_zero{}};
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(pred(1)), bool>);
  CHECK(pred(1));
  CHECK(!pred(0));
}

TEST_CASE("WrappedFunctionVoidUnary", "[wrapped_function]")
{
  int acc = 0;
  const thrust::detail::wrapped_function<void_unary, void> f{void_unary{&acc}};
  STATIC_REQUIRE(::cuda::std::is_void_v<decltype(f(1))>);
  f(3);
  CHECK(acc == 3);

  const thrust::detail::wrapped_function<int_identity, void> discard{int_identity{}};
  STATIC_REQUIRE(::cuda::std::is_void_v<decltype(discard(1))>);
  discard(1);
}

TEST_CASE("WrappedFunctionDeducedNonVoid", "[wrapped_function]")
{
  const thrust::detail::wrapped_function<int_minus> op{int_minus{}};
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(op(5, 2)), int>);
  CHECK(op(5, 2) == 3);
}

TEST_CASE("WrappedFunctionExplicitResultConversion", "[wrapped_function]")
{
  const thrust::detail::wrapped_function<int_identity, float> op{int_identity{}};
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(op(42)), float>);
  CHECK(op(42) == 42.0f);
}
