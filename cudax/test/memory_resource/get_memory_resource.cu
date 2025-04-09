//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <cuda/experimental/memory_resource.cuh>

#include "test_resource.cuh"
#include <testing.cuh>

using device_resource = cuda::experimental::device_memory_resource;

struct with_get_resource_const_lvalue
{
  device_resource res_{};

  const device_resource& get_memory_resource() const noexcept
  {
    return res_;
  }
};
C2H_TEST("Can call get_memory_resource on a type with a get_memory_resource method that returns a const lvalue",
         "[resource]")
{
  with_get_resource_const_lvalue val{};
  auto&& res = ::cuda::experimental::get_memory_resource(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), const device_resource&>);
  CUDAX_CHECK(val.res_ == res);
}

struct with_get_resource_rvalue
{
  device_resource res_{};

  device_resource get_memory_resource() const noexcept
  {
    return res_;
  }
};
C2H_TEST("Can call get_memory_resource on a type with a get_memory_resource method returns an rvalue", "[resource]")
{
  with_get_resource_rvalue val{};
  auto&& res = ::cuda::experimental::get_memory_resource(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), device_resource&&>);
  CUDAX_CHECK(val.res_ == res);
}

struct with_get_resource_non_const
{
  device_resource res_{};

  device_resource get_memory_resource() noexcept
  {
    return res_;
  }
};
C2H_TEST("Cannot call get_memory_resource on a type with a non-const get_memory_resource method", "[resource]")
{
  STATIC_REQUIRE(
    !::cuda::std::is_invocable_v<::cuda::experimental::get_memory_resource_t, const with_get_resource_non_const&>);
}

struct env_with_query_const_ref
{
  device_resource res_{};

  const device_resource& query(::cuda::experimental::get_memory_resource_t) const noexcept
  {
    return res_;
  }
};
C2H_TEST("Can call get_memory_resource on an env with a get_memory_resource query that returns a const lvalue",
         "[resource]")
{
  env_with_query_const_ref val{};
  auto&& res = ::cuda::experimental::get_memory_resource(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), const device_resource&>);
  CUDAX_CHECK(val.res_ == res);
}

struct env_with_query_rvalue
{
  device_resource res_{};

  device_resource query(::cuda::experimental::get_memory_resource_t) const noexcept
  {
    return res_;
  }
};
C2H_TEST("Can call get_memory_resource on an env with a get_memory_resource query that returns an rvalue", "[resource]")
{
  env_with_query_rvalue val{};
  auto&& res = ::cuda::experimental::get_memory_resource(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), device_resource&&>);
  CUDAX_CHECK(val.res_ == res);
}

struct env_with_query_non_const
{
  device_resource res_{};

  const device_resource& query(::cuda::experimental::get_memory_resource_t) noexcept
  {
    return res_;
  }
};
C2H_TEST("Cannot call get_memory_resource on an env with a non-const query", "[resource]")
{
  STATIC_REQUIRE(
    !::cuda::std::is_invocable_v<::cuda::experimental::get_memory_resource_t, const env_with_query_non_const&>);
}

struct env_with_query_and_method
{
  device_resource res_{};

  const device_resource& get_memory_resource() const noexcept
  {
    return res_;
  }

  device_resource query(::cuda::experimental::get_memory_resource_t) const noexcept
  {
    return res_;
  }
};
C2H_TEST("Can call get_memory_resource on a type with both get_memory_resource and query", "[resource]")
{
  env_with_query_and_method val{};
  auto&& res = ::cuda::experimental::get_memory_resource(val);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(res), const device_resource&>);
  CUDAX_CHECK(val.res_ == res);
}

struct with_get_resource_non_async
{
  struct resource
  {
    void* allocate(std::size_t, std::size_t)
    {
      return nullptr;
    }

    void deallocate(void*, std::size_t, std::size_t) noexcept {}

    bool operator==(const resource&) const noexcept
    {
      return true;
    }
    bool operator!=(const resource&) const noexcept
    {
      return false;
    }
  };
  resource res_{};

  resource get_memory_resource() const noexcept
  {
    return res_;
  }
};
C2H_TEST("Cannot call get_memory_resource on an env with a non-async resource", "[resource]")
{
  STATIC_REQUIRE(
    !::cuda::std::is_invocable_v<::cuda::experimental::get_memory_resource_t, const with_get_resource_non_async&>);
}
