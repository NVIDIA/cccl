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

#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/coop/__group/coalesced_group.cuh>
#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/generic_group.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/group_by.cuh>
#include <cuda/experimental/coop/__group/synchronizer/lane_synchronizer.cuh>
#include <cuda/experimental/coop/__group/this_group.cuh>
#include <cuda/experimental/coop/__group/virtual_group.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <template <class...> class _Lhs, template <class...> class _Rhs>
inline constexpr bool __is_same_templ_v = false;
template <template <class...> class _Templ>
inline constexpr bool __is_same_templ_v<_Templ, _Templ> = true;

// exhaustive make_group

_CCCL_TEMPLATE(template <class...> class _GroupTempl,
               class... _Args,
               class _Group = decltype(_GroupTempl{::cuda::std::declval<_Args>()...}))
_CCCL_REQUIRES(group<_Group>)
[[nodiscard]] _CCCL_DEVICE_API _Group
make_group(_Args&&... __args) noexcept(::cuda::std::is_nothrow_constructible_v<_Group, _Args...>)
{
  return _Group{::cuda::std::forward<_Args>(__args)...};
}

// non-exhaustive make_group

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

// CG interoperability

#  if _CCCL_HAS_COOPERATIVE_GROUPS()

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_group(const ::cooperative_groups::grid_group&, const _HierarchyLike& __hier_like) noexcept
{
  return this_grid{__hier_like};
}

#    if defined(_CG_HAS_CLUSTER_GROUP)
_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_group(const ::cooperative_groups::cluster_group&, const _HierarchyLike& __hier_like) noexcept
{
  return this_cluster{__hier_like};
}
#    endif // _CG_HAS_CLUSTER_GROUP

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_group(const ::cooperative_groups::thread_block&, const _HierarchyLike& __hier_like) noexcept
{
  return this_block{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike, class _CgGroup = ::cooperative_groups::coalesced_group)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_group(const ::cuda::std::type_identity_t<_CgGroup>& __cg_group, const _HierarchyLike& __hier_like) noexcept
{
  // todo(dabayer): CG's coalesced_group may be dynamically tiled, thus it's not exactly equivalent to CG's
  // coalesced_group. We should probably just return an ordinary generic_group that would be able to handle this.
  coalesced_group __ret{__hier_like};
  _CCCL_ASSERT(gpu_thread.count(__ret) == __cg_group.size(), "CG coalesced group can't be partitioned for now");
  return __ret;
}

_CCCL_TEMPLATE(unsigned _Size, class _ParentT, class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_group(const ::cooperative_groups::thread_block_tile<_Size, _ParentT>&, const _HierarchyLike& __hier_like) noexcept
{
  if constexpr (_Size == 1 && ::cuda::std::is_same_v<_ParentT, void>)
  {
    return this_thread{__hier_like};
  }
  else if constexpr (_Size < 32 && ::cuda::std::is_same_v<_ParentT, ::cooperative_groups::thread_block>)
  {
    return generic_group{
      gpu_thread,
      this_warp{__hier_like},
      group_by{::cuda::std::integral_constant<::cuda::std::size_t, _Size>{}},
      lane_synchronizer{}};
  }
  else if constexpr (_Size == 32 && ::cuda::std::is_same_v<_ParentT, ::cooperative_groups::thread_block>)
  {
    return this_warp{__hier_like};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_ParentT>,
                  "make_group has not been implemented for this thread_block_tile CG group yet");
  }
}

#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAKE_GROUP_CUH
