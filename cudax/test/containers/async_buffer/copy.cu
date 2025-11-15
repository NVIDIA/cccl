//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory_resource/shared_resource.h>
#include <cuda/memory_resource>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"

#if _CCCL_CTK_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::host_accessible>,
                                  cuda::std::tuple<unsigned long long, cuda::mr::device_accessible>,
                                  cuda::std::tuple<int, cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 6) ^^^ / _CCCL_CTK_BELOW(12, 6) vvv
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::device_accessible>>;
#endif // ^^^ _CCCL_CTK_BELOW(12, 6) ^^^

template <class T1, class T2, class... PropertiesSuperSet, class... PropertiesSubset>
constexpr bool is_matching_buffer(const cudax::buffer<T1, PropertiesSuperSet...>&,
                                  const cudax::buffer<T2, PropertiesSubset...>&) noexcept
{
  return ::cuda::std::__type_set_contains_v<::cuda::std::__make_type_set<PropertiesSuperSet...>, PropertiesSubset...>
      && ::cuda::std::is_same_v<T1, T2>;
}

C2H_CCCLRT_TEST("cudax::buffer make_buffer", "[container][buffer]", test_types)
{
  using TestT    = c2h::get<0, TestType>;
  using Resource = typename extract_properties<TestT>::resource;
  using Buffer   = typename extract_properties<TestT>::buffer;
  using T        = typename Buffer::value_type;

  cudax::stream stream{cuda::device_ref{0}};
  Resource resource = extract_properties<TestT>::get_resource();

  using MatchingResource = typename extract_properties<TestT>::matching_resource;

  SECTION("Same resource and stream")
  {
    { // empty input
      const Buffer input{stream, resource};
      const Buffer buf = cudax::make_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{stream, resource};
      const Buffer buf = cudax::make_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_buffer(input.stream(), input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different stream")
  {
    cudax::stream other_stream{cuda::device_ref{0}};
    { // empty input
      const Buffer input{stream, resource};
      const Buffer buf = cudax::make_buffer(other_stream, input.memory_resource(), input);
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const Buffer buf = cudax::make_buffer(other_stream, input.memory_resource(), input);
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource and stream")
  {
    cudax::stream other_stream{cuda::device_ref{0}};
    { // empty input
      const Buffer input{stream, resource};
      auto buf = cudax::make_buffer(other_stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_buffer(other_stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }

  SECTION("Different resource, same stream")
  {
    { // empty input
      const Buffer input{stream, resource};
      const auto buf = cudax::make_buffer(stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_buffer(stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }

    { // empty input
      const Buffer input{stream, resource};
      const auto buf = cudax::make_buffer(stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(buf.empty());
      CUDAX_CHECK(buf.data() == nullptr);
    }

    { // non-empty input
      const Buffer input{stream, resource, {T(1), T(42), T(1337), T(0), T(12), T(-1)}};
      const auto buf = cudax::make_buffer(stream, resource, input);
      static_assert(is_matching_buffer(buf, input));
      CUDAX_CHECK(!buf.empty());
      CUDAX_CHECK(equal_range(buf));
    }
  }
}

C2H_CCCLRT_TEST("make_buffer variants", "[container][buffer]")
{
  cudax::stream stream{cuda::device_ref{0}};
  const cudax::buffer<int, cuda::mr::device_accessible, other_property> input{
    stream,
    cuda::device_default_memory_pool(cuda::device_ref{0}),
    {int(1), int(42), int(1337), int(0), int(12), int(-1)}};

  // straight from a resource
  auto buf =
    cuda::experimental::make_buffer(input.stream(), cuda::device_default_memory_pool(cuda::device_ref{0}), input);
  CUDAX_CHECK(equal_range(buf));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, cuda::mr::device_accessible>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, cuda::mr::host_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf)::__resource_t, other_property>);

  auto buf2 = cuda::experimental::make_buffer<int, cuda::mr::device_accessible>(
    input.stream(), cuda::device_default_memory_pool(cuda::device_ref{0}), input);
  CUDAX_CHECK(equal_range(buf2));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf2)::__resource_t, cuda::mr::host_accessible>);

  // from any resource
  auto any_res = cuda::mr::any_resource<cuda::mr::device_accessible, other_property>(
    cuda::device_default_memory_pool(cuda::device_ref{0}));
  auto buf3 = cudax::make_buffer(input.stream(), any_res, input);
  CUDAX_CHECK(equal_range(buf3));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, cuda::mr::device_accessible>);
  static_assert(::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf3)::__resource_t, cuda::mr::host_accessible>);

  auto buf4 = cudax::make_buffer<int, cuda::mr::device_accessible>(input.stream(), any_res, input);
  CUDAX_CHECK(equal_range(buf4));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf4)::__resource_t, cuda::mr::host_accessible>);

  // from a resource reference
  auto res_ref = cuda::mr::resource_ref<cuda::mr::device_accessible, other_property>{any_res};
  auto buf5    = cudax::make_buffer(input.stream(), res_ref, input);
  CUDAX_CHECK(equal_range(buf5));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, cuda::mr::device_accessible>);
  static_assert(::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf5)::__resource_t, cuda::mr::host_accessible>);

  auto buf6 = cudax::make_buffer<int, cuda::mr::device_accessible>(input.stream(), res_ref, input);
  CUDAX_CHECK(equal_range(buf6));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf6)::__resource_t, cuda::mr::host_accessible>);

  auto shared_res =
    cuda::mr::make_shared_resource<cuda::device_memory_pool_ref>(cuda::device_default_memory_pool(cuda::device_ref{0}));
  auto buf7 = cudax::make_buffer(input.stream(), shared_res, input);
  CUDAX_CHECK(equal_range(buf7));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf7)::__resource_t, cuda::mr::host_accessible>);

  auto buf8 = cudax::make_buffer<int, cuda::mr::device_accessible>(input.stream(), shared_res, input);
  CUDAX_CHECK(equal_range(buf8));
  static_assert(
    ::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, cuda::mr::device_accessible>);
  static_assert(!::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, other_property>);
  static_assert(
    !::cuda::mr::synchronous_resource_with<typename decltype(buf8)::__resource_t, cuda::mr::host_accessible>);
}

C2H_CCCLRT_TEST("make_buffer with legacy resource", "[container][buffer]")
{
  cudax::stream stream{cuda::device_ref{0}};
  auto resource = cuda::legacy_pinned_memory_resource{};
  cudax::buffer<int, cuda::mr::host_accessible> input{
    stream, resource, {int(1), int(42), int(1337), int(0), int(12), int(-1)}};
  auto buf = cudax::make_buffer(input.stream(), resource, input);
  CUDAX_CHECK(equal_range(buf));
}
