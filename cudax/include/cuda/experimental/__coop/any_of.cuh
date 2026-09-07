//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___COOP_ANY_OF_CUH
#define _CUDA_EXPERIMENTAL___COOP_ANY_OF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__numeric/reduce.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/optional>

#include <cuda/experimental/__utility/result_policy.cuh>
#include <cuda/experimental/group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo: Can we make any_of be implemented as reduce(group, data, cuda::std::logical_or{})?

namespace cuda::experimental::coop
{
template <bool _Dummy = false>
_CCCL_DEVICE_API auto __any_of_impl(...)
{
  static_assert(_Dummy, "cudax::coop::any_of is not supported for the group");
}

template <bool _Broadcasted, class _Hierarchy>
[[nodiscard]] _CCCL_DEVICE_API auto
__any_of_impl(::cuda::std::bool_constant<_Broadcasted>, const this_thread<_Hierarchy>&, bool __thread_data)
{
  if constexpr (_Broadcasted)
  {
    return __thread_data;
  }
  else
  {
    return ::cuda::std::optional{__thread_data};
  }
}

_CCCL_TEMPLATE(bool _Broadcasted, class _Group)
_CCCL_REQUIRES(is_group<_Group> _CCCL_AND ::cuda::std::is_same_v<typename _Group::level_type, warp_level>)
[[nodiscard]] _CCCL_DEVICE_API auto
__any_of_impl(::cuda::std::bool_constant<_Broadcasted>, const _Group& __group, bool __thread_data) noexcept
{
  const auto& __mapping_result = __group.__mapping_result();
  const auto __result          = static_cast<bool>(::__any_sync(__mapping_result.lane_mask().value(), __thread_data));
  if constexpr (_Broadcasted)
  {
    return __result;
  }
  else
  {
    return (gpu_thread.is_root_rank(__group)) ? ::cuda::std::optional{__result} : ::cuda::std::nullopt;
  }
}

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, bool>)
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<bool> any_of(const _Group& __group, _Tp __thread_data)
{
  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::any_of");
  return ::cuda::experimental::coop::__any_of_impl(::cuda::std::false_type{}, __group, __thread_data);
}

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, bool>)
[[nodiscard]] _CCCL_DEVICE_API bool any_of(broadcasted_t, const _Group& __group, _Tp __thread_data)
{
  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::any_of");
  return ::cuda::experimental::coop::__any_of_impl(::cuda::std::true_type{}, __group, __thread_data);
}

_CCCL_TEMPLATE(class _Group, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, bool>)
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<bool> any_of(const _Group& __group, _Tp (&__thread_data)[_Np])
{
  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::any_of");
  return ::cuda::experimental::coop::any_of(
    __group, ::cuda::std::reduce(__thread_data, __thread_data + _Np, false, ::cuda::std::logical_or<bool>{}));
}

_CCCL_TEMPLATE(class _Group, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES(::cuda::std::is_same_v<_Tp, bool>)
[[nodiscard]] _CCCL_DEVICE_API bool any_of(broadcasted_t, const _Group& __group, _Tp (&__thread_data)[_Np])
{
  _CCCL_ASSERT(gpu_thread.is_part_of(__group), "Only threads that are part of the group can call cudax::coop::any_of");
  return ::cuda::experimental::coop::any_of(
    broadcasted,
    __group,
    ::cuda::std::reduce(__thread_data, __thread_data + _Np, false, ::cuda::std::logical_or<bool>{}));
}

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES((!::cuda::std::is_same_v<_Tp, bool>) )
auto any_of(const _Group& __group, _Tp __thread_data) = delete;

_CCCL_TEMPLATE(class _Group, class _Tp)
_CCCL_REQUIRES((!::cuda::std::is_same_v<_Tp, bool>) )
auto any_of(broadcasted_t, const _Group& __group, _Tp __thread_data) = delete;

_CCCL_TEMPLATE(class _Group, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES((!::cuda::std::is_same_v<_Tp, bool>) )
auto any_of(const _Group& __group, _Tp (&__thread_data)[_Np]) = delete;

_CCCL_TEMPLATE(class _Group, class _Tp, ::cuda::std::size_t _Np)
_CCCL_REQUIRES((!::cuda::std::is_same_v<_Tp, bool>) )
auto any_of(broadcasted_t, const _Group& __group, _Tp (&__thread_data)[_Np]) = delete;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___COOP_ANY_OF_CUH
