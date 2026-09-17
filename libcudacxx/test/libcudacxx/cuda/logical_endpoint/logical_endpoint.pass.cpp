//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/logical_endpoint>

// UNSUPPORTED: nvrtc
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda/logical_endpoint>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

template <class _Tp, class = void>
struct has_add_device : cuda::std::false_type
{};

template <class _Tp>
struct has_add_device<
  _Tp,
  cuda::std::void_t<decltype(cuda::std::declval<const _Tp&>().add_device(cuda::std::declval<cuda::device_ref>()))>>
    : cuda::std::true_type
{};

template <class _Tp, class = void>
struct has_is_ready : cuda::std::false_type
{};

template <class _Tp>
struct has_is_ready<_Tp, cuda::std::void_t<decltype(cuda::std::declval<const _Tp&>().is_ready())>>
    : cuda::std::true_type
{};

template <class _Tp, class = void>
struct has_wait_ready_for : cuda::std::false_type
{};

template <class _Tp>
struct has_wait_ready_for<
  _Tp,
  cuda::std::void_t<decltype(cuda::std::declval<const _Tp&>().wait_ready_for(cuda::std::chrono::nanoseconds{1}))>>
    : cuda::std::true_type
{};

static_assert(cuda::std::is_trivially_copyable_v<cuda::logical_endpoint_id>);
static_assert(cuda::std::is_trivially_copyable_v<cuda::unicast_logical_endpoint_ref>);
static_assert(cuda::std::is_trivially_copyable_v<cuda::multicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_default_constructible_v<cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_default_constructible_v<cuda::unicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_default_constructible_v<cuda::multicast_logical_endpoint_ref>);
static_assert(cuda::std::is_default_constructible_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::unicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::multicast_logical_endpoint_ref>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::std::uint32_t>);
static_assert(
  !cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(!has_add_device<cuda::unicast_logical_endpoint_ref>::value);
static_assert(has_add_device<cuda::multicast_logical_endpoint_ref>::value);
static_assert(!has_is_ready<cuda::logical_endpoint_id_range>::value);
static_assert(!has_wait_ready_for<cuda::logical_endpoint_id_range>::value);

TEST_FUNC constexpr bool test_endpoint_ids()
{
  cuda::logical_endpoint_id id{7};
  cuda::unicast_logical_endpoint_ref unicast_ref{cuda::logical_endpoint_id{7}};
  cuda::multicast_logical_endpoint_ref multicast_ref{cuda::logical_endpoint_id{7}};
  cuda::unicast_logical_endpoint_ref unicast_ref_from_id{id};
  cuda::multicast_logical_endpoint_ref multicast_ref_from_id{id};
  cuda::logical_endpoint_id advanced = id;
  advanced += 5;
  cuda::logical_endpoint_id retreated = advanced;
  retreated -= 3;

  assert(id.native_handle() == 7);
  assert((id + 5).native_handle() == 12);
  assert((5 + id).native_handle() == 12);
  assert((advanced - 3).native_handle() == 9);
  assert(advanced.native_handle() == 12);
  assert(retreated.native_handle() == 9);
  assert(unicast_ref.id() == id);
  assert(multicast_ref.id() == id);
  assert(unicast_ref_from_id.id() == id);
  assert(multicast_ref_from_id.id() == id);

  return true;
}

bool test_empty_id_ranges()
{
  cuda::logical_endpoint_id_range ids;
  assert(ids.size() == 0);

  cuda::logical_endpoint_id_range moved{cuda::std::move(ids)};
  assert(ids.size() == 0);
  assert(moved.size() == 0);

  cuda::logical_endpoint_id_range assigned;
  assigned = cuda::std::move(moved);
  assert(assigned.size() == 0);
  assert(moved.size() == 0);

  return true;
}

static_assert(test_endpoint_ids());

#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)
  assert(test_endpoint_ids());
  assert(test_empty_id_ranges());
#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

  return 0;
}
