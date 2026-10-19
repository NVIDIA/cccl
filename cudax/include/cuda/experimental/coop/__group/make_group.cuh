//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAKE_GROUP_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAKE_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/generic_group.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/virtual_group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <template <class...> class _Lhs, template <class...> class _Rhs>
inline constexpr bool __is_same_templ_v = false;
template <template <class...> class _Templ>
inline constexpr bool __is_same_templ_v<_Templ, _Templ> = true;

_CCCL_TEMPLATE(template <class...> class _GroupTempl,
               class... _Args,
               class _Group = decltype(_GroupTempl{::cuda::std::declval<_Args>()...}))
_CCCL_REQUIRES(group<_Group>)
[[nodiscard]] _CCCL_DEVICE_API _Group
make_group(_Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<_Group, _Args...>)
{
  return _Group{::cuda::std::forward<_Args>(__args)...};
}

_CCCL_TEMPLATE(
  template <class...> class _GroupTempl,
  class _Unit,
  class _ParentGroup,
  class _Mapping,
  class _Synchronizer,
  class _MappingResult        = __group_mapping_result_t<non_exhaustive_t, _Unit, _ParentGroup, _Mapping>,
  class _SynchronizerInstance = __group_synchronizer_instance_t<_Unit, _ParentGroup, _MappingResult, _Synchronizer>,
  class _Group                = _GroupTempl<_Unit, _ParentGroup, _MappingResult, _SynchronizerInstance>)
_CCCL_REQUIRES(__is_same_templ_v<generic_group, _GroupTempl> _CCCL_AND __is_hierarchy_level_v<_Unit> _CCCL_AND
                 group<_ParentGroup> _CCCL_AND __unit_same_as_or_below_v<_Unit, typename _ParentGroup::unit_type>)
[[nodiscard]] _CCCL_DEVICE_API ::cuda::std::optional<_Group> make_group(
  const non_exhaustive_t& __non_exhaustive,
  const _Unit& __unit,
  const _ParentGroup& __parent,
  _Mapping&& __mapping,
  _Synchronizer&& __synchronizer) noexcept
{
  const auto __mapping_result = ::cuda::experimental::coop::__do_group_mapping(
    __non_exhaustive, __unit, __parent, ::cuda::std::forward<_Mapping>(__mapping));
  using _Ret = ::cuda::std::optional<_Group>;
  return (__mapping_result.is_valid())
         ? _Ret{::cuda::std::in_place,
                __non_exhaustive,
                __unit,
                __parent,
                __mapping_result,
                ::cuda::std::forward<_Synchronizer>(__synchronizer)}
         : _Ret{};
}

_CCCL_TEMPLATE(template <class...> class _GroupTempl, class... _Args)
_CCCL_REQUIRES(__is_same_templ_v<virtual_group, _GroupTempl>)
auto make_group(const non_exhaustive_t&, _Args&&... __args) = delete;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAKE_GROUP_CUH
