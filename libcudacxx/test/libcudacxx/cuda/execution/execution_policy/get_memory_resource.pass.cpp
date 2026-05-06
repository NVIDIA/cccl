//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// UNSUPPORTED: nvrtc

#include <cuda/functional>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/std/memory>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include "test_macros.h"

_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored

struct test_resource
{
  TEST_FUNC void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }

  TEST_FUNC void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  TEST_FUNC void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return &_val;
  }

  TEST_FUNC void deallocate(cuda::stream_ref, void* ptr, std::size_t, std::size_t)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  TEST_FUNC bool operator==(const test_resource& other) const
  {
    return _val == other._val;
  }
  TEST_FUNC bool operator!=(const test_resource& other) const
  {
    return _val != other._val;
  }

  friend constexpr void get_property(const test_resource&, ::cuda::mr::device_accessible) noexcept {}

  int _val = 0;
};
static_assert(::cuda::mr::resource<test_resource>);

template <class Policy>
void test(Policy pol)
{
  auto old_stream        = cuda::__call_or(::cuda::get_stream, cuda::stream_ref{cudaStreamPerThread}, pol);
  auto fallback_resource = ::cuda::device_default_memory_pool(cuda::device_ref{0});
  { // Ensure that the plain policy is not callable with get_memory_resource
    assert(cuda::__call_or(::cuda::mr::get_memory_resource, fallback_resource, pol) == fallback_resource);
  }

  { // Ensure that we can attach a memory resource to an execution policy
    test_resource resource{42};
    auto pol_with_resource = pol.with(cuda::mr::get_memory_resource, resource);
    assert(cuda::mr::get_memory_resource(pol_with_resource) == resource);
    assert(cuda::__call_or(::cuda::get_stream, cuda::stream_ref{cudaStreamPerThread}, pol_with_resource) == old_stream);

    using policy_t = decltype(pol_with_resource);
    static_assert(cuda::std::is_execution_policy_v<policy_t>);
  }

  { // Ensure that attaching a memory resource multiple times just overwrites the old one
    test_resource resource{42};
    auto pol_with_resource = pol.with(cuda::mr::get_memory_resource, resource);
    assert(cuda::mr::get_memory_resource(pol_with_resource) == resource);
    assert(cuda::__call_or(::cuda::get_stream, cuda::stream_ref{cudaStreamPerThread}, pol_with_resource) == old_stream);

    test_resource other_resource{1337};
    decltype(auto) pol_with_other_resource = pol_with_resource.with(cuda::mr::get_memory_resource, other_resource);

    // The original resource is unchanged
    assert(cuda::mr::get_memory_resource(pol_with_resource) == resource);
    assert(cuda::mr::get_memory_resource(pol_with_other_resource) == other_resource);
    assert(cuda::__call_or(::cuda::get_stream, cuda::stream_ref{cudaStreamPerThread}, pol_with_resource) == old_stream);
  }
}

void test()
{
  namespace execution = cuda::std::execution;
  static_assert(!execution::__queryable_with<execution::sequenced_policy, ::cuda::mr::get_memory_resource_t>);
  static_assert(!execution::__queryable_with<execution::parallel_policy, ::cuda::mr::get_memory_resource_t>);
  static_assert(
    !execution::__queryable_with<execution::parallel_unsequenced_policy, ::cuda::mr::get_memory_resource_t>);
  static_assert(!execution::__queryable_with<execution::unsequenced_policy, ::cuda::mr::get_memory_resource_t>);

  test(cuda::execution::gpu);

  // Ensure that all works even if we have a stream attached
  ::cuda::stream stream{cuda::device_ref{0}};
  test(cuda::execution::gpu.with(cuda::get_stream, stream));
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))

  return 0;
}
