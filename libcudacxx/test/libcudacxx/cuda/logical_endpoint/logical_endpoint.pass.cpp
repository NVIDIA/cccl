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

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda/launch>
#include <cuda/logical_endpoint>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/optional>
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
struct has_bitwise_or : cuda::std::false_type
{};

template <class _Tp>
struct has_bitwise_or<_Tp, cuda::std::void_t<decltype(cuda::std::declval<_Tp>() | cuda::std::declval<_Tp>())>>
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
struct has_wait_until_ready : cuda::std::false_type
{};

template <class _Tp>
struct has_wait_until_ready<_Tp, cuda::std::void_t<decltype(cuda::std::declval<const _Tp&>().wait_until_ready())>>
    : cuda::std::true_type
{};

static_assert(cuda::std::is_trivially_copyable_v<cuda::logical_endpoint_id>);
static_assert(cuda::std::is_trivially_copyable_v<cuda::unicast_logical_endpoint_ref>);
static_assert(cuda::std::is_trivially_copyable_v<cuda::multicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_default_constructible_v<cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_default_constructible_v<cuda::unicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_default_constructible_v<cuda::multicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_default_constructible_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_default_constructible_v<cuda::unicast_logical_endpoint>);
static_assert(cuda::std::is_default_constructible_v<cuda::multicast_logical_endpoint>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::unicast_logical_endpoint_ref>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::multicast_logical_endpoint_ref>);
static_assert(cuda::std::is_constructible_v<cuda::unicast_logical_endpoint_spec, cuda::device_ref>);
static_assert(cuda::std::is_constructible_v<cuda::multicast_logical_endpoint_spec, unsigned int>);
static_assert(!cuda::std::is_constructible_v<cuda::multicast_logical_endpoint_spec, cuda::device_ref, unsigned int>);
static_assert(!cuda::std::is_copy_constructible_v<cuda::unicast_logical_endpoint>);
static_assert(!cuda::std::is_copy_constructible_v<cuda::multicast_logical_endpoint>);
static_assert(cuda::std::is_move_constructible_v<cuda::unicast_logical_endpoint>);
static_assert(cuda::std::is_move_constructible_v<cuda::multicast_logical_endpoint>);
static_assert(cuda::std::is_move_assignable_v<cuda::unicast_logical_endpoint>);
static_assert(cuda::std::is_move_assignable_v<cuda::multicast_logical_endpoint>);
static_assert(!cuda::std::is_constructible_v<bool, const cuda::unicast_logical_endpoint&>);
static_assert(!cuda::std::is_constructible_v<bool, const cuda::multicast_logical_endpoint&>);
static_assert(cuda::std::is_convertible_v<const cuda::unicast_logical_endpoint&, cuda::unicast_logical_endpoint_ref>);
static_assert(
  cuda::std::is_convertible_v<const cuda::multicast_logical_endpoint&, cuda::multicast_logical_endpoint_ref>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::std::uint32_t>);
static_assert(
  !cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(cuda::std::is_constructible_v<cuda::unicast_logical_endpoint,
                                            const cuda::unicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<cuda::unicast_logical_endpoint,
                                            cuda::logical_endpoint_id,
                                            const cuda::unicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<cuda::unicast_logical_endpoint,
                                            const cuda::logical_endpoint_id_range&,
                                            cuda::std::uint32_t,
                                            const cuda::unicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<cuda::multicast_logical_endpoint,
                                            const cuda::multicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<cuda::multicast_logical_endpoint,
                                            cuda::logical_endpoint_id,
                                            const cuda::multicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<cuda::multicast_logical_endpoint,
                                            const cuda::logical_endpoint_id_range&,
                                            cuda::std::uint32_t,
                                            const cuda::multicast_logical_endpoint_spec&,
                                            cuda::std::uint64_t>);
static_assert(!cuda::std::is_constructible_v<cuda::multicast_logical_endpoint, cuda::device_ref, cuda::std::uint64_t>);

static_assert(cuda::std::is_same_v<cuda::transformed_device_argument_t<cuda::unicast_logical_endpoint&>,
                                   cuda::unicast_logical_endpoint_ref>);
static_assert(cuda::std::is_same_v<cuda::transformed_device_argument_t<cuda::multicast_logical_endpoint&>,
                                   cuda::multicast_logical_endpoint_ref>);
static_assert(cuda::std::is_same_v<
              decltype(cuda::std::declval<cuda::unicast_logical_endpoint&>().release()),
              cuda::std::pair<cuda::logical_endpoint_id, cuda::std::optional<cuda::logical_endpoint_id_range>>>);
static_assert(cuda::std::is_same_v<
              decltype(cuda::std::declval<cuda::multicast_logical_endpoint&>().release()),
              cuda::std::pair<cuda::logical_endpoint_id, cuda::std::optional<cuda::logical_endpoint_id_range>>>);
static_assert(!has_add_device<cuda::unicast_logical_endpoint_ref>::value);
static_assert(has_add_device<cuda::multicast_logical_endpoint_ref>::value);
static_assert(!has_add_device<cuda::unicast_logical_endpoint>::value);
static_assert(has_add_device<cuda::multicast_logical_endpoint>::value);
static_assert(has_bitwise_or<cuda::logical_endpoint_flag>::value);
static_assert(!has_bitwise_or<cuda::logical_endpoint_ipc_handle_type>::value);
static_assert(!has_is_ready<cuda::logical_endpoint_id_range>::value);
static_assert(!has_wait_until_ready<cuda::logical_endpoint_id_range>::value);

TEST_FUNC constexpr bool test_endpoint_ids()
{
  cuda::logical_endpoint_id id = 7;
  cuda::unicast_logical_endpoint_ref unicast_ref{7};
  cuda::multicast_logical_endpoint_ref multicast_ref{7};
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

constexpr bool test_endpoint_flags()
{
  const cuda::logical_endpoint_flag flags =
    cuda::logical_endpoint_flag::none | cuda::logical_endpoint_flag::counted_ops;

  assert(static_cast<unsigned>(flags) == static_cast<unsigned>(cuda::logical_endpoint_flag::counted_ops));

  return true;
}

bool test_empty_owning_endpoints()
{
  cuda::unicast_logical_endpoint unicast;
  cuda::multicast_logical_endpoint multicast;

  assert(!unicast.has_value());
  assert(!multicast.has_value());
  assert(unicast.size() == 0);
  assert(multicast.size() == 0);
  assert(unicast.bind_alignment() == 0);
  assert(multicast.bind_alignment() == 0);

  cuda::unicast_logical_endpoint moved_unicast{cuda::std::move(unicast)};
  cuda::multicast_logical_endpoint moved_multicast{cuda::std::move(multicast)};
  assert(!unicast.has_value());
  assert(!multicast.has_value());
  assert(!moved_unicast.has_value());
  assert(!moved_multicast.has_value());

  cuda::unicast_logical_endpoint assigned_unicast;
  cuda::multicast_logical_endpoint assigned_multicast;
  assigned_unicast   = cuda::std::move(moved_unicast);
  assigned_multicast = cuda::std::move(moved_multicast);
  assert(!assigned_unicast.has_value());
  assert(!assigned_multicast.has_value());
  assert(!moved_unicast.has_value());
  assert(!moved_multicast.has_value());

  return true;
}

static_assert(test_endpoint_ids());
static_assert(test_endpoint_flags());

#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)
  assert(test_endpoint_ids());
  assert(test_endpoint_flags());
  assert(test_empty_owning_endpoints());
#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

  return 0;
}
