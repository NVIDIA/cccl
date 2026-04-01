//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/execution>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include "test_macros.h"

struct SomeValue
{
  int value;
};

struct SomeProperty
{
  _CCCL_TEMPLATE(class Env)
  _CCCL_REQUIRES(cuda::std::execution::__queryable_with<const Env&, SomeProperty>)
  [[nodiscard]] constexpr SomeValue operator()(const Env& env) const
  {
    return env.query(SomeProperty{});
  }
};

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

  friend constexpr void get_property(const test_resource&, ::cuda::mr::device_accessible) noexcept {}

  int _val = 0;
};
static_assert(::cuda::mr::resource<test_resource>);

template <class Policy>
void test(const Policy& policy)
{
  { // the policy can take a stream
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(cuda::get_stream, stream);
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a stream_ref
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(cuda::get_stream, ::cuda::stream_ref{stream});
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a cudaStream_t
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(cuda::get_stream, stream.get());
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a stream_ref as an environment
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(cuda::stream_ref{stream});
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a cuda::stream as an environment
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(stream);
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a cudaStream_t as an environment
    cuda::stream stream{cuda::device_ref{0}};
    static_assert(!cuda::std::__is_callable_v<cuda::get_stream_t, const Policy&>);
    const auto new_policy = policy.with(stream.get());
    static_assert(cuda::std::__is_callable_v<cuda::get_stream_t, decltype(new_policy)>);
    auto&& result = cuda::get_stream(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), cuda::stream_ref&&>);
    assert(stream == result);
  }

  { // the policy can take a memory resource by lvalue
    test_resource resource{};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(cuda::mr::get_memory_resource, resource);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a memory resource by prvalue -> need to own
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(cuda::mr::get_memory_resource, test_resource{});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::any_resource<>&>);
  }

  { // the policy can take a resource_ref by lvalue
    test_resource resource{};
    cuda::mr::resource_ref<::cuda::mr::device_accessible> resource_ref{resource};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(cuda::mr::get_memory_resource, resource_ref);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a prvalue resource_ref -> remains resource_ref
    test_resource resource{};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy =
      policy.with(cuda::mr::get_memory_resource, cuda::mr::resource_ref<::cuda::mr::device_accessible>{resource});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
  }

  { // the policy can take a cuda::mr::any_resource by lvalue
    cuda::mr::any_resource<::cuda::mr::device_accessible> resource{test_resource{}};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(cuda::mr::get_memory_resource, resource);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a prvalue cuda::mr::any_resource -> needs to own
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(
      cuda::mr::get_memory_resource, cuda::mr::any_resource<::cuda::mr::device_accessible>{test_resource{}});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::any_resource<>&>);
  }

  { // the policy can take a memory resource by lvalue as an environment
    test_resource resource{};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(resource);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a memory resource by prvalue as an environment -> need to own
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(test_resource{});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::any_resource<>&>);
  }

  { // the policy can take a resource_ref by lvalue as an environment
    test_resource resource{};
    cuda::mr::resource_ref<::cuda::mr::device_accessible> resource_ref{resource};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(resource_ref);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a prvalue resource_ref as an environment -> remains resource_ref
    test_resource resource{};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(cuda::mr::resource_ref<::cuda::mr::device_accessible>{resource});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
  }

  { // the policy can take a cuda::mr::any_resource by lvalue as an environment
    cuda::mr::any_resource<::cuda::mr::device_accessible> resource{test_resource{}};
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(resource);
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::resource_ref<>&>);
    assert(resource == result);
  }

  { // the policy can take a prvalue cuda::mr::any_resource as an environment -> needs to own
    static_assert(!cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, const Policy&>);
    const auto new_policy = policy.with(
      cuda::mr::get_memory_resource, cuda::mr::any_resource<::cuda::mr::device_accessible>{test_resource{}});
    static_assert(cuda::std::__is_callable_v<cuda::mr::get_memory_resource_t, decltype(new_policy)>);
    auto&& result = cuda::mr::get_memory_resource(new_policy);
    static_assert(cuda::std::is_same_v<decltype(result), const cuda::mr::any_resource<>&>);
  }

  { // the policy can take an arbitrary tag that is queryable
    const SomeProperty property{};
    static_assert(!cuda::std::__is_callable_v<const SomeProperty&, const Policy&>);
    const auto new_policy = policy.with(property, SomeValue{42});
    static_assert(cuda::std::__is_callable_v<const SomeProperty&, decltype(new_policy)>);
    auto&& result = property(new_policy);
    assert(result.value == 42);
  }
}

bool test()
{
  test(cuda::std::execution::seq);
  test(cuda::std::execution::par);
  test(cuda::std::execution::par_unseq);
  test(cuda::std::execution::unseq);

  // Cuda specific execution policy
  test(cuda::execution::gpu);

  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))

  return 0;
}
