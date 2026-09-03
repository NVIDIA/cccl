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
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cstdio>
#include <stdexcept>

#include <cuda_runtime_api.h>

#include "test_macros.h"

#if _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

constexpr int driver_version_13_3 = 13030;

using unicast_ref_t        = cuda::unicast_logical_endpoint_ref;
using multicast_ref_t      = cuda::multicast_logical_endpoint_ref;
using unicast_endpoint_t   = cuda::unicast_logical_endpoint;
using multicast_endpoint_t = cuda::multicast_logical_endpoint;
using unicast_spec_t       = cuda::unicast_logical_endpoint_spec;
using multicast_spec_t     = cuda::multicast_logical_endpoint_spec;
using le_release_t = cuda::std::pair<cuda::logical_endpoint_id, cuda::std::optional<cuda::logical_endpoint_id_range>>;
using handle_t     = cuda::logical_endpoint_handle;
using le_ipc_t     = cuda::logical_endpoint_ipc_handle_type;
using le_flag_t    = cuda::logical_endpoint_flag;
using le_limits_t  = cuda::logical_endpoint_limits;

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
static_assert(cuda::std::is_trivially_copyable_v<unicast_ref_t>);
static_assert(cuda::std::is_trivially_copyable_v<multicast_ref_t>);
static_assert(cuda::std::is_trivially_copyable_v<handle_t>);
static_assert(!cuda::std::is_default_constructible_v<cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_default_constructible_v<unicast_ref_t>);
static_assert(!cuda::std::is_default_constructible_v<multicast_ref_t>);
static_assert(!cuda::std::is_default_constructible_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_default_constructible_v<unicast_endpoint_t>);
static_assert(cuda::std::is_default_constructible_v<multicast_endpoint_t>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(cuda::std::is_convertible_v<cuda::std::uint32_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_constructible_v<unicast_ref_t, cuda::std::uint32_t>);
static_assert(cuda::std::is_constructible_v<multicast_ref_t, cuda::std::uint32_t>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, unicast_ref_t>);
static_assert(!cuda::std::is_convertible_v<cuda::std::uint32_t, multicast_ref_t>);
static_assert(cuda::std::is_constructible_v<unicast_ref_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_constructible_v<multicast_ref_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_constructible_v<unicast_spec_t, cuda::device_ref>);
static_assert(cuda::std::is_constructible_v<unicast_spec_t, cuda::device_ref, le_flag_t>);
static_assert(cuda::std::is_constructible_v<unicast_spec_t, cuda::device_ref, le_flag_t, le_ipc_t>);
static_assert(cuda::std::is_constructible_v<multicast_spec_t, unsigned int>);
static_assert(cuda::std::is_constructible_v<multicast_spec_t, unsigned int, le_flag_t>);
static_assert(cuda::std::is_constructible_v<multicast_spec_t, unsigned int, le_flag_t, le_ipc_t>);
static_assert(!cuda::std::is_constructible_v<multicast_spec_t, cuda::device_ref, unsigned int>);
static_assert(!cuda::std::is_constructible_v<unicast_endpoint_t, cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_constructible_v<multicast_endpoint_t, cuda::logical_endpoint_id>);
static_assert(!cuda::std::is_copy_constructible_v<unicast_endpoint_t>);
static_assert(!cuda::std::is_copy_constructible_v<multicast_endpoint_t>);
static_assert(cuda::std::is_move_constructible_v<unicast_endpoint_t>);
static_assert(cuda::std::is_move_constructible_v<multicast_endpoint_t>);
static_assert(cuda::std::is_move_assignable_v<unicast_endpoint_t>);
static_assert(cuda::std::is_move_assignable_v<multicast_endpoint_t>);
static_assert(!cuda::std::is_constructible_v<bool, const unicast_endpoint_t&>);
static_assert(!cuda::std::is_constructible_v<bool, const multicast_endpoint_t&>);
static_assert(cuda::std::is_convertible_v<const unicast_endpoint_t&, unicast_ref_t>);
static_assert(cuda::std::is_convertible_v<const multicast_endpoint_t&, multicast_ref_t>);
static_assert(cuda::std::is_copy_constructible_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_move_constructible_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_copy_assignable_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_move_assignable_v<cuda::logical_endpoint_id_range>);
static_assert(cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::std::uint32_t>);
static_assert(
  !cuda::std::is_constructible_v<cuda::logical_endpoint_id_range, cuda::logical_endpoint_id, cuda::std::uint32_t>);
static_assert(cuda::std::is_constructible_v<unicast_endpoint_t, const unicast_spec_t&, cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<unicast_endpoint_t, const handle_t&>);
static_assert(
  cuda::std::
    is_constructible_v<unicast_endpoint_t, cuda::logical_endpoint_id, const unicast_spec_t&, cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<unicast_endpoint_t, cuda::logical_endpoint_id, const handle_t&>);
static_assert(cuda::std::is_constructible_v<unicast_endpoint_t,
                                            const cuda::logical_endpoint_id_range&,
                                            cuda::std::uint32_t,
                                            const unicast_spec_t&,
                                            cuda::std::uint64_t>);
static_assert(
  cuda::std::
    is_constructible_v<unicast_endpoint_t, const cuda::logical_endpoint_id_range&, cuda::std::uint32_t, const handle_t&>);
static_assert(cuda::std::is_constructible_v<multicast_endpoint_t, const multicast_spec_t&, cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<multicast_endpoint_t, const handle_t&>);
static_assert(
  cuda::std::
    is_constructible_v<multicast_endpoint_t, cuda::logical_endpoint_id, const multicast_spec_t&, cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<multicast_endpoint_t, cuda::logical_endpoint_id, const handle_t&>);
static_assert(cuda::std::is_constructible_v<multicast_endpoint_t,
                                            const cuda::logical_endpoint_id_range&,
                                            cuda::std::uint32_t,
                                            const multicast_spec_t&,
                                            cuda::std::uint64_t>);
static_assert(cuda::std::is_constructible_v<multicast_endpoint_t,
                                            const cuda::logical_endpoint_id_range&,
                                            cuda::std::uint32_t,
                                            const handle_t&>);
static_assert(!cuda::std::is_constructible_v<multicast_endpoint_t, cuda::device_ref, cuda::std::uint64_t>);

using unicast_spec_device_t         = decltype(cuda::std::declval<const unicast_spec_t&>().device());
using unicast_spec_flags_t          = decltype(cuda::std::declval<const unicast_spec_t&>().flags());
using unicast_spec_ipc_t            = decltype(cuda::std::declval<const unicast_spec_t&>().ipc_handle_type());
using multicast_spec_count_t        = decltype(cuda::std::declval<const multicast_spec_t&>().num_devices());
using multicast_spec_flags_t        = decltype(cuda::std::declval<const multicast_spec_t&>().flags());
using multicast_spec_ipc_t          = decltype(cuda::std::declval<const multicast_spec_t&>().ipc_handle_type());
using unicast_ref_bind_t            = decltype(cuda::std::declval<const unicast_ref_t&>().bind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, static_cast<void*>(nullptr), cuda::std::uint64_t{}));
using multicast_ref_bind_t          = decltype(cuda::std::declval<const multicast_ref_t&>().bind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, static_cast<void*>(nullptr), cuda::std::uint64_t{}));
using unicast_ref_bind_mem_t        = decltype(cuda::std::declval<const unicast_ref_t&>().bind(
  cuda::std::declval<cuda::device_ref>(),
  cuda::std::uint64_t{},
  cuda::std::declval<CUmemGenericAllocationHandle>(),
  cuda::std::uint64_t{},
  cuda::std::uint64_t{}));
using multicast_ref_bind_mem_t      = decltype(cuda::std::declval<const multicast_ref_t&>().bind(
  cuda::std::declval<cuda::device_ref>(),
  cuda::std::uint64_t{},
  cuda::std::declval<CUmemGenericAllocationHandle>(),
  cuda::std::uint64_t{},
  cuda::std::uint64_t{}));
using unicast_endpoint_bind_t       = decltype(cuda::std::declval<const unicast_endpoint_t&>().bind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, static_cast<void*>(nullptr), cuda::std::uint64_t{}));
using multicast_endpoint_bind_t     = decltype(cuda::std::declval<const multicast_endpoint_t&>().bind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, static_cast<void*>(nullptr), cuda::std::uint64_t{}));
using unicast_endpoint_bind_mem_t   = decltype(cuda::std::declval<const unicast_endpoint_t&>().bind(
  cuda::std::declval<cuda::device_ref>(),
  cuda::std::uint64_t{},
  cuda::std::declval<CUmemGenericAllocationHandle>(),
  cuda::std::uint64_t{},
  cuda::std::uint64_t{},
  unsigned{}));
using multicast_endpoint_bind_mem_t = decltype(cuda::std::declval<const multicast_endpoint_t&>().bind(
  cuda::std::declval<cuda::device_ref>(),
  cuda::std::uint64_t{},
  cuda::std::declval<CUmemGenericAllocationHandle>(),
  cuda::std::uint64_t{},
  cuda::std::uint64_t{},
  unsigned{}));
using unicast_ref_unbind_t          = decltype(cuda::std::declval<const unicast_ref_t&>().unbind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, cuda::std::uint64_t{}));
using multicast_ref_unbind_t        = decltype(cuda::std::declval<const multicast_ref_t&>().unbind(
  cuda::std::declval<cuda::device_ref>(), cuda::std::uint64_t{}, cuda::std::uint64_t{}));
using unicast_ref_wait_t            = decltype(cuda::std::declval<const unicast_ref_t&>().wait_until_ready());
using multicast_ref_wait_t          = decltype(cuda::std::declval<const multicast_ref_t&>().wait_until_ready());
using unicast_ref_wait_timeout_t =
  decltype(cuda::std::declval<const unicast_ref_t&>().wait_until_ready(cuda::std::chrono::nanoseconds{}));
using multicast_ref_wait_timeout_t =
  decltype(cuda::std::declval<const multicast_ref_t&>().wait_until_ready(cuda::std::chrono::nanoseconds{}));
using unicast_endpoint_export_t   = decltype(cuda::std::declval<const unicast_endpoint_t&>().export_handle());
using multicast_endpoint_export_t = decltype(cuda::std::declval<const multicast_endpoint_t&>().export_handle());
using unicast_endpoint_export_ipc_t =
  decltype(cuda::std::declval<const unicast_endpoint_t&>().export_handle(le_ipc_t::fabric));
using multicast_endpoint_export_ipc_t =
  decltype(cuda::std::declval<const multicast_endpoint_t&>().export_handle(le_ipc_t::fabric));
using unicast_endpoint_release_t     = decltype(cuda::std::declval<unicast_endpoint_t&>().release());
using multicast_endpoint_release_t   = decltype(cuda::std::declval<multicast_endpoint_t&>().release());
using unicast_endpoint_has_value_t   = decltype(cuda::std::declval<const unicast_endpoint_t&>().has_value());
using multicast_endpoint_has_value_t = decltype(cuda::std::declval<const multicast_endpoint_t&>().has_value());
using unicast_endpoint_wait_t        = decltype(cuda::std::declval<const unicast_endpoint_t&>().wait_until_ready());
using multicast_endpoint_wait_t      = decltype(cuda::std::declval<const multicast_endpoint_t&>().wait_until_ready());
using release_range_t                = decltype(cuda::std::declval<le_release_t&>().second);
using unicast_limits_t               = decltype(cuda::std::declval<const unicast_spec_t&>().limits());
using multicast_limits_t             = decltype(cuda::std::declval<const multicast_spec_t&>().limits());
using unicast_is_supported_t         = decltype(cuda::std::declval<const unicast_spec_t&>().is_supported());
using unicast_is_supported_with_device_t =
  decltype(cuda::std::declval<const unicast_spec_t&>().is_supported(cuda::std::declval<cuda::device_ref>()));
using multicast_is_supported_t =
  decltype(cuda::std::declval<const multicast_spec_t&>().is_supported(cuda::std::declval<cuda::device_ref>()));
using unicast_launch_transform_t   = cuda::transformed_device_argument_t<unicast_endpoint_t&>;
using multicast_launch_transform_t = cuda::transformed_device_argument_t<multicast_endpoint_t&>;
using range_size_t                 = decltype(cuda::std::declval<const cuda::logical_endpoint_id_range&>().size());
using range_base_id_t              = decltype(cuda::std::declval<const cuda::logical_endpoint_id_range&>().base_id());
using range_subscript_t = decltype(cuda::std::declval<const cuda::logical_endpoint_id_range&>()[cuda::std::uint32_t{}]);
using range_release_t   = decltype(cuda::std::declval<cuda::logical_endpoint_id_range&>().release());
using id_plus_t         = decltype(cuda::std::declval<cuda::logical_endpoint_id>() + cuda::std::uint32_t{});
using id_rplus_t        = decltype(cuda::std::uint32_t{} + cuda::std::declval<cuda::logical_endpoint_id>());
using id_minus_t        = decltype(cuda::std::declval<cuda::logical_endpoint_id>() - cuda::std::uint32_t{});
using id_plus_eq_t      = decltype(cuda::std::declval<cuda::logical_endpoint_id&>() += cuda::std::uint32_t{});
using id_minus_eq_t     = decltype(cuda::std::declval<cuda::logical_endpoint_id&>() -= cuda::std::uint32_t{});
using id_plus_literal_t = decltype(cuda::std::declval<cuda::logical_endpoint_id>() + 1);
using id_rplus_literal_t = decltype(1 + cuda::std::declval<cuda::logical_endpoint_id>());
using id_minus_literal_t = decltype(cuda::std::declval<cuda::logical_endpoint_id>() - 1);

__global__ void
kernel_accepts_logical_endpoint_refs(unicast_ref_t unicast, multicast_ref_t multicast, ::CUlogicalEndpointId* ids)
{
  ids[0] = unicast.native_handle();
  ids[1] = multicast.id().native_handle();
}

static_assert(cuda::std::is_same_v<unicast_spec_device_t, cuda::device_ref>);
static_assert(cuda::std::is_same_v<unicast_spec_flags_t, le_flag_t>);
static_assert(cuda::std::is_same_v<unicast_spec_ipc_t, le_ipc_t>);
static_assert(cuda::std::is_same_v<multicast_spec_count_t, unsigned int>);
static_assert(cuda::std::is_same_v<multicast_spec_flags_t, le_flag_t>);
static_assert(cuda::std::is_same_v<multicast_spec_ipc_t, le_ipc_t>);
static_assert(cuda::std::is_same_v<unicast_ref_bind_t, void>);
static_assert(cuda::std::is_same_v<multicast_ref_bind_t, void>);
static_assert(cuda::std::is_same_v<unicast_ref_bind_mem_t, void>);
static_assert(cuda::std::is_same_v<multicast_ref_bind_mem_t, void>);
static_assert(cuda::std::is_same_v<unicast_endpoint_bind_t, void>);
static_assert(cuda::std::is_same_v<multicast_endpoint_bind_t, void>);
static_assert(cuda::std::is_same_v<unicast_endpoint_bind_mem_t, void>);
static_assert(cuda::std::is_same_v<multicast_endpoint_bind_mem_t, void>);
static_assert(cuda::std::is_same_v<unicast_ref_unbind_t, void>);
static_assert(cuda::std::is_same_v<multicast_ref_unbind_t, void>);
static_assert(cuda::std::is_same_v<unicast_ref_wait_t, bool>);
static_assert(cuda::std::is_same_v<multicast_ref_wait_t, bool>);
static_assert(cuda::std::is_same_v<unicast_ref_wait_timeout_t, bool>);
static_assert(cuda::std::is_same_v<multicast_ref_wait_timeout_t, bool>);
static_assert(cuda::std::is_same_v<unicast_endpoint_export_t, handle_t>);
static_assert(cuda::std::is_same_v<multicast_endpoint_export_t, handle_t>);
static_assert(cuda::std::is_same_v<unicast_endpoint_export_ipc_t, handle_t>);
static_assert(cuda::std::is_same_v<multicast_endpoint_export_ipc_t, handle_t>);
static_assert(cuda::std::is_same_v<unicast_endpoint_release_t, le_release_t>);
static_assert(cuda::std::is_same_v<multicast_endpoint_release_t, le_release_t>);
static_assert(cuda::std::is_same_v<unicast_endpoint_has_value_t, bool>);
static_assert(cuda::std::is_same_v<multicast_endpoint_has_value_t, bool>);
static_assert(cuda::std::is_same_v<unicast_endpoint_wait_t, bool>);
static_assert(cuda::std::is_same_v<multicast_endpoint_wait_t, bool>);
static_assert(cuda::std::is_same_v<release_range_t, cuda::std::optional<cuda::logical_endpoint_id_range>>);
static_assert(cuda::std::is_same_v<unicast_limits_t, le_limits_t>);
static_assert(cuda::std::is_same_v<multicast_limits_t, le_limits_t>);
static_assert(cuda::std::is_same_v<unicast_is_supported_t, bool>);
static_assert(cuda::std::is_same_v<unicast_is_supported_with_device_t, bool>);
static_assert(cuda::std::is_same_v<multicast_is_supported_t, bool>);
static_assert(cuda::std::is_same_v<unicast_launch_transform_t, unicast_ref_t>);
static_assert(cuda::std::is_same_v<multicast_launch_transform_t, multicast_ref_t>);
static_assert(cuda::std::is_same_v<range_size_t, cuda::std::uint32_t>);
static_assert(cuda::std::is_same_v<range_base_id_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<range_subscript_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<range_release_t, cuda::std::pair<cuda::logical_endpoint_id, cuda::std::uint32_t>>);
static_assert(cuda::std::is_same_v<id_plus_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<id_rplus_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<id_minus_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<id_plus_eq_t, cuda::logical_endpoint_id&>);
static_assert(cuda::std::is_same_v<id_minus_eq_t, cuda::logical_endpoint_id&>);
static_assert(cuda::std::is_same_v<id_plus_literal_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<id_rplus_literal_t, cuda::logical_endpoint_id>);
static_assert(cuda::std::is_same_v<id_minus_literal_t, cuda::logical_endpoint_id>);
static_assert(!has_add_device<unicast_ref_t>::value);
static_assert(has_add_device<multicast_ref_t>::value);
static_assert(!has_add_device<unicast_endpoint_t>::value);
static_assert(has_add_device<multicast_endpoint_t>::value);
static_assert(has_bitwise_or<le_flag_t>::value);
static_assert(!has_bitwise_or<le_ipc_t>::value);
static_assert(!has_is_ready<cuda::logical_endpoint_id_range>::value);
static_assert(!has_wait_until_ready<cuda::logical_endpoint_id_range>::value);

TEST_FUNC constexpr bool test_endpoint_ids()
{
  cuda::logical_endpoint_id id = 7;
  unicast_ref_t unicast_ref{7};
  multicast_ref_t multicast_ref{7};
  unicast_ref_t unicast_ref_from_id{id};
  multicast_ref_t multicast_ref_from_id{id};
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
  const le_flag_t flags = le_flag_t::none | le_flag_t::counted_ops;

  assert(static_cast<unsigned>(flags) == static_cast<unsigned>(le_flag_t::counted_ops));
  assert(static_cast<unsigned>(le_ipc_t::none) == CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_NONE);
  assert(static_cast<unsigned>(le_ipc_t::fabric) == CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC);

  return true;
}

bool test_endpoint_specs()
{
  unicast_spec_t unicast{cuda::device_ref{0}, le_flag_t::counted_ops, le_ipc_t::fabric};
  multicast_spec_t multicast{4, le_flag_t::none, le_ipc_t::none};

  assert(unicast.device() == cuda::device_ref{0});
  assert(unicast.flags() == le_flag_t::counted_ops);
  assert(unicast.ipc_handle_type() == le_ipc_t::fabric);
  assert(multicast.num_devices() == 4);
  assert(multicast.flags() == le_flag_t::none);
  assert(multicast.ipc_handle_type() == le_ipc_t::none);

  return true;
}

bool test_fabric_handle_native_handle()
{
  handle_t handle{};

  assert(handle.native_handle() != nullptr);
  assert(static_cast<const handle_t&>(handle).native_handle() != nullptr);

  return true;
}

bool test_import_rejects_untyped_fabric_handle()
{
#  if TEST_HAS_EXCEPTIONS()
  handle_t handle{};

  bool caught_unicast = false;
  try
  {
    unicast_endpoint_t endpoint{cuda::logical_endpoint_id{13}, handle};
    assert(false);
    (void) endpoint;
  }
  catch (const ::std::invalid_argument&)
  {
    caught_unicast = true;
  }

  bool caught_multicast = false;
  try
  {
    multicast_endpoint_t endpoint{cuda::logical_endpoint_id{17}, handle};
    assert(false);
    (void) endpoint;
  }
  catch (const ::std::invalid_argument&)
  {
    caught_multicast = true;
  }

  assert(caught_unicast);
  assert(caught_multicast);
#  endif // TEST_HAS_EXCEPTIONS()

  return true;
}

bool test_null_pointer_bind_is_rejected_before_driver_call()
{
#  if TEST_HAS_EXCEPTIONS()
  unicast_ref_t unicast_ref{cuda::logical_endpoint_id{19}};
  multicast_ref_t multicast_ref{cuda::logical_endpoint_id{23}};

  bool caught_unicast = false;
  try
  {
    unicast_ref.bind(cuda::device_ref{0}, 0, nullptr, 16);
    assert(false);
  }
  catch (const ::std::invalid_argument&)
  {
    caught_unicast = true;
  }

  bool caught_multicast = false;
  try
  {
    multicast_ref.bind(cuda::device_ref{0}, 0, nullptr, 16);
    assert(false);
  }
  catch (const ::std::invalid_argument&)
  {
    caught_multicast = true;
  }

  assert(caught_unicast);
  assert(caught_multicast);
#  endif // TEST_HAS_EXCEPTIONS()

  return true;
}

bool test_empty_owning_endpoints()
{
  unicast_endpoint_t unicast;
  multicast_endpoint_t multicast;

  assert(!unicast.has_value());
  assert(!multicast.has_value());
  assert(unicast.size() == 0);
  assert(multicast.size() == 0);
  assert(unicast.bind_alignment() == 0);
  assert(multicast.bind_alignment() == 0);

  unicast_endpoint_t moved_unicast{cuda::std::move(unicast)};
  multicast_endpoint_t moved_multicast{cuda::std::move(multicast)};
  assert(!unicast.has_value());
  assert(!multicast.has_value());
  assert(!moved_unicast.has_value());
  assert(!moved_multicast.has_value());

  unicast_endpoint_t assigned_unicast;
  multicast_endpoint_t assigned_multicast;
  assigned_unicast   = cuda::std::move(moved_unicast);
  assigned_multicast = cuda::std::move(moved_multicast);
  assert(!assigned_unicast.has_value());
  assert(!assigned_multicast.has_value());
  assert(!moved_unicast.has_value());
  assert(!moved_multicast.has_value());

  return true;
}

bool logical_endpoint_runtime_is_supported()
{
  int driver_version = 0;
  if (::cudaDriverGetVersion(&driver_version) != cudaSuccess)
  {
    std::fprintf(stderr, "skipping: CUDA driver version could not be queried\n");
    return false;
  }
  if (driver_version < driver_version_13_3)
  {
    std::fprintf(stderr, "skipping: logical endpoints require a CUDA 13.3 driver\n");
    return false;
  }

  int device_count = 0;
  if (::cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0)
  {
    std::fprintf(stderr, "skipping: logical endpoint tests require a CUDA device\n");
    return false;
  }

  cuda::device_ref device{0};
  unicast_spec_t unicast{device};
  multicast_spec_t multicast{1};
  if (!unicast.is_supported(device))
  {
    std::fprintf(stderr, "skipping: unicast logical endpoints are not supported\n");
    return false;
  }
  if (!multicast.is_supported(device))
  {
    std::fprintf(stderr, "skipping: multicast logical endpoints are not supported\n");
    return false;
  }

  return true;
}

static_assert(test_endpoint_ids());
static_assert(test_endpoint_flags());

#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

int main(int, char**)
{
#if _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)
  if (!logical_endpoint_runtime_is_supported())
  {
    return 0;
  }
  assert(test_endpoint_ids());
  assert(test_endpoint_flags());
  assert(test_endpoint_specs());
  assert(test_fabric_handle_native_handle());
  assert(test_import_rejects_untyped_fabric_handle());
  assert(test_null_pointer_bind_is_rejected_before_driver_call());
  assert(test_empty_owning_endpoints());
#endif // _CCCL_CTK_AT_LEAST(13, 3) && !TEST_COMPILER(NVRTC)

  return 0;
}
