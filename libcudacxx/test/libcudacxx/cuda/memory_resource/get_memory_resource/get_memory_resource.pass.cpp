//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/__functional/call_or.h>
#include <cuda/memory_resource>
#include <cuda/std/type_traits>

struct test_resource
{
  __host__ __device__ void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }

  __host__ __device__ void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  __host__ __device__ void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return &_val;
  }

  __host__ __device__ void deallocate(cuda::stream_ref, void* ptr, std::size_t, std::size_t)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  __host__ __device__ bool operator==(const test_resource& other) const
  {
    return _val == other._val;
  }
  __host__ __device__ bool operator!=(const test_resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;
};

__host__ __device__ void test()
{
  test_resource invalid_resource{42};
  { // Can call get_memory_resource on a type with a get_memory_resource method that returns a const lvalue
    struct with_get_resource_const_lvalue
    {
      test_resource res_{};

      __host__ __device__ const test_resource& get_memory_resource() const noexcept
      {
        return res_;
      }
    };
    with_get_resource_const_lvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);

    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(val.res_ == res_query);
  }

  { // Can call get_memory_resource on a type with a get_memory_resource method returns an rvalue
    struct with_get_resource_rvalue
    {
      test_resource res_{};

      __host__ __device__ test_resource get_memory_resource() const noexcept
      {
        return res_;
      }
    };
    with_get_resource_rvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), test_resource&&>);
    assert(val.res_ == res);

    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(val.res_ == res_query);
  }

  { // Cannot call get_memory_resource on a type with a non-const get_memory_resource method
    struct with_get_resource_non_const
    {
      test_resource res_{};

      __host__ __device__ test_resource get_memory_resource() noexcept
      {
        return res_;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const with_get_resource_non_const&>);

    with_get_resource_non_const val{};
    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(res_query == invalid_resource);
  }

  { // Can call get_memory_resource on an env with a get_memory_resource query that returns a const lvalue
    struct env_with_query_const_ref
    {
      test_resource res_{};

      __host__ __device__ const test_resource& query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };
    env_with_query_const_ref val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);

    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(val.res_ == res_query);
  }

  { // Can call get_memory_resource on an env with a get_memory_resource query that returns an rvalue
    struct env_with_query_rvalue
    {
      test_resource res_{};

      __host__ __device__ test_resource query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };

    env_with_query_rvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), test_resource&&>);
    assert(val.res_ == res);

    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(val.res_ == res_query);
  }

  { // Cannot call get_memory_resource on an env with a non-const query
    struct env_with_query_non_const
    {
      test_resource res_{};

      __host__ __device__ const test_resource& query(::cuda::mr::get_memory_resource_t) noexcept
      {
        return res_;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const env_with_query_non_const&>);

    env_with_query_non_const val{};
    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(res_query == invalid_resource);
  }

  { // Can call get_memory_resource on a type with both get_memory_resource and query
    struct env_with_query_and_method
    {
      test_resource res_{};

      __host__ __device__ const test_resource& get_memory_resource() const noexcept
      {
        return res_;
      }

      __host__ __device__ test_resource query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };

    env_with_query_and_method val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);

    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(val.res_ == res_query);
  }

  { // Cannot call get_memory_resource on an env with a non-async resource
    struct with_get_resource_non_async
    {
      struct resource
      {
        __host__ __device__ void* allocate_sync(std::size_t, std::size_t)
        {
          return nullptr;
        }

        __host__ __device__ void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}

        __host__ __device__ bool operator==(const resource&) const noexcept
        {
          return true;
        }

        __host__ __device__ bool operator!=(const resource&) const noexcept
        {
          return false;
        }
      };
      resource res_{};

      __host__ __device__ resource get_memory_resource() const noexcept
      {
        return res_;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const with_get_resource_non_async&>);

    with_get_resource_non_async val{};
    auto res_query = ::cuda::__call_or(::cuda::mr::get_memory_resource, invalid_resource, val);
    assert(res_query == invalid_resource);
  }
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
