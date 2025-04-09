//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>

#include <cuda/experimental/execution.cuh>

#include <testing.cuh>

using cuda::experimental::execution::execution_policy;

struct with_get_execution_policy_const_lvalue
{
  execution_policy pol_ = execution_policy::sequenced_host;

  const execution_policy& get_execution_policy() const noexcept
  {
    return pol_;
  }
};
C2H_TEST("Can call get_execution_policy on a type with a get_execution_policy method that returns a const lvalue",
         "[execution, policies]")
{
  with_get_execution_policy_const_lvalue val{};
  auto&& res = cuda::experimental::execution::get_execution_policy(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), execution_policy&&>);
  CHECK(val.pol_ == res);
}

struct with_get_execution_policy_rvalue
{
  execution_policy pol_{};

  execution_policy get_execution_policy() const noexcept
  {
    return pol_;
  }
};
C2H_TEST("Can call get_execution_policy on a type with a get_execution_policy method returns an rvalue",
         "[execution, policies]")
{
  with_get_execution_policy_rvalue val{};
  auto&& res = cuda::experimental::execution::get_execution_policy(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), execution_policy&&>);
  CHECK(val.pol_ == res);
}

struct with_get_execution_policy_non_const
{
  execution_policy pol_{};

  execution_policy get_execution_policy() noexcept
  {
    return pol_;
  }
};
C2H_TEST("Cannot call get_execution_policy on a type with a non-const get_execution_policy method",
         "[execution, policies]")
{
  STATIC_REQUIRE(!::cuda::std::is_invocable_v<cuda::experimental::execution::get_execution_policy_t,
                                              const with_get_execution_policy_non_const&>);
}

struct env_with_query_const_ref
{
  execution_policy pol_{};

  execution_policy query(cuda::experimental::execution::get_execution_policy_t) const noexcept
  {
    return pol_;
  }
};
C2H_TEST("Can call get_execution_policy on an env with a get_execution_policy query that returns a const lvalue",
         "[execution, policies]")
{
  env_with_query_const_ref val{};
  auto&& res = cuda::experimental::execution::get_execution_policy(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), execution_policy&&>);
  CHECK(val.pol_ == res);
}

struct env_with_query_rvalue
{
  execution_policy pol_{};

  execution_policy query(cuda::experimental::execution::get_execution_policy_t) const noexcept
  {
    return pol_;
  }
};
C2H_TEST("Can call get_execution_policy on an env with a get_execution_policy query that returns an rvalue",
         "[execution, policies]")
{
  env_with_query_rvalue val{};
  auto&& res = cuda::experimental::execution::get_execution_policy(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), execution_policy&&>);
  CHECK(val.pol_ == res);
}

struct env_with_query_non_const
{
  execution_policy pol_{};

  execution_policy query(cuda::experimental::execution::get_execution_policy_t) noexcept
  {
    return pol_;
  }
};
C2H_TEST("Cannot call get_execution_policy on an env with a non-const query", "[execution, policies]")
{
  STATIC_REQUIRE(
    !::cuda::std::is_invocable_v<cuda::experimental::execution::get_execution_policy_t, const env_with_query_non_const&>);
}

struct env_with_query_and_method
{
  execution_policy pol_{};

  execution_policy get_execution_policy() const noexcept
  {
    return pol_;
  }

  execution_policy query(cuda::experimental::execution::get_execution_policy_t) const noexcept
  {
    return pol_;
  }
};
C2H_TEST("Can call get_execution_policy on a type with both get_execution_policy and query", "[execution, policies]")
{
  env_with_query_and_method val{};
  auto&& res = cuda::experimental::execution::get_execution_policy(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), execution_policy&&>);
  CHECK(val.pol_ == res);
}
