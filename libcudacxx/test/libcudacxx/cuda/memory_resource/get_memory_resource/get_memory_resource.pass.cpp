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

#include <cuda/memory_resource>
#include <cuda/std/type_traits>

struct test_resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const test_resource& other) const
  {
    return _val == other._val;
  }
  bool operator!=(const test_resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;
};

void test()
{
  { // Can call get_memory_resource on a type with a get_memory_resource method that returns a const lvalue
    struct with_get_resource_const_lvalue
    {
      test_resource res_{};

      const test_resource& get_memory_resource() const noexcept
      {
        return res_;
      }
    };
    with_get_resource_const_lvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);
  }

  { // Can call get_memory_resource on a type with a get_memory_resource method returns an rvalue
    struct with_get_resource_rvalue
    {
      test_resource res_{};

      test_resource get_memory_resource() const noexcept
      {
        return res_;
      }
    };
    with_get_resource_rvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), test_resource&&>);
    assert(val.res_ == res);
  }

  { // Cannot call get_memory_resource on a type with a non-const get_memory_resource method
    struct with_get_resource_non_const
    {
      test_resource res_{};

      test_resource get_memory_resource() noexcept
      {
        return res_;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const with_get_resource_non_const&>);
  }

  { // Can call get_memory_resource on an env with a get_memory_resource query that returns a const lvalue
    struct env_with_query_const_ref
    {
      test_resource res_{};

      const test_resource& query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };
    env_with_query_const_ref val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);
  }

  { // Can call get_memory_resource on an env with a get_memory_resource query that returns an rvalue
    struct env_with_query_rvalue
    {
      test_resource res_{};

      test_resource query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };

    env_with_query_rvalue val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), test_resource&&>);
    assert(val.res_ == res);
  }

  { // Cannot call get_memory_resource on an env with a non-const query
    struct env_with_query_non_const
    {
      test_resource res_{};

      const test_resource& query(::cuda::mr::get_memory_resource_t) noexcept
      {
        return res_;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const env_with_query_non_const&>);
  }

  { // Can call get_memory_resource on a type with both get_memory_resource and query
    struct env_with_query_and_method
    {
      test_resource res_{};

      const test_resource& get_memory_resource() const noexcept
      {
        return res_;
      }

      test_resource query(::cuda::mr::get_memory_resource_t) const noexcept
      {
        return res_;
      }
    };

    env_with_query_and_method val{};
    auto&& res = ::cuda::mr::get_memory_resource(val);
    static_assert(cuda::std::is_same_v<decltype(res), const test_resource&>);
    assert(val.res_ == res);
  }

  { // Cannot call get_memory_resource on an env with a non-async resource
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
    static_assert(!::cuda::std::is_invocable_v<::cuda::mr::get_memory_resource_t, const with_get_resource_non_async&>);
  }
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, test();)

  return 0;
}
