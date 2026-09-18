//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_CG_INTEROP_CUH
#define _CUDA_EXPERIMENTAL___GROUP_CG_INTEROP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if __has_include(<cooperative_groups.h>) && !defined(_CUDAX_DISABLE_CG_INTEROP)

#  include <cuda/hierarchy>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/type_identity.h>

#  include <cuda/experimental/__group/coalesced_group.cuh>
#  include <cuda/experimental/__group/generic_group.cuh>
#  include <cuda/experimental/__group/mapping/group_by.cuh>
#  include <cuda/experimental/__group/synchronizer/lane_synchronizer.cuh>
#  include <cuda/experimental/__group/this_group.cuh>

// We don't need to include the whole <cooperative_groups.h> header, we can just forward declare the CG group classes in
// the CG namespace. However, the CG namespace is versioned, so we still need to include the minimal header.
#  include <cooperative_groups/details/info.h>
#  include <cuda/std/__cccl/prologue.h>

#  if !defined(_CCCL_DOXYGEN_INVOKED)

_CG_BEGIN_NAMESPACE

class thread_group;
class grid_group;
#    if defined(_CG_HAS_CLUSTER_GROUP)
class cluster_group;
#    endif // _CG_HAS_CLUSTER_GROUP
class thread_block;
class coalesced_group;
template <unsigned _Size, typename _ParentT>
class thread_block_tile;

_CG_END_NAMESPACE

namespace cuda::experimental
{
_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_cg_equivalent_group(const ::cooperative_groups::grid_group&, const _HierarchyLike& __hier_like) noexcept
{
  return this_grid{__hier_like};
}

#    if defined(_CG_HAS_CLUSTER_GROUP)
_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_cg_equivalent_group(const ::cooperative_groups::cluster_group&, const _HierarchyLike& __hier_like) noexcept
{
  return this_cluster{__hier_like};
}
#    endif // _CG_HAS_CLUSTER_GROUP

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto
make_cg_equivalent_group(const ::cooperative_groups::thread_block&, const _HierarchyLike& __hier_like) noexcept
{
  return this_block{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike, class _CgGroup = ::cooperative_groups::coalesced_group)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto make_cg_equivalent_group(
  const ::cuda::std::type_identity_t<_CgGroup>& __cg_group, const _HierarchyLike& __hier_like) noexcept
{
  // todo(dabayer): CG's coalesced_group may be dynamically tiled, thus it's not exactly equivalent to CG's
  // coalesced_group. We should probably just return an ordinary generic_group that would be able to handle this.
  coalesced_group __ret{__hier_like};
  _CCCL_ASSERT(gpu_thread.count(__ret) == __cg_group.size(), "CG coalesced group can't be partitioned for now");
  return __ret;
}

_CCCL_TEMPLATE(unsigned _Size, class _ParentT, class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto make_cg_equivalent_group(
  const ::cooperative_groups::thread_block_tile<_Size, _ParentT>&, const _HierarchyLike& __hier_like) noexcept
{
  if constexpr (_Size == 1 && ::cuda::std::is_same_v<_ParentT, void>)
  {
    return this_thread{__hier_like};
  }
  else if constexpr (_Size < 32 && ::cuda::std::is_same_v<_ParentT, ::cooperative_groups::thread_block>)
  {
    return generic_group{gpu_thread, this_warp{__hier_like}, group_by<_Size>{}, lane_synchronizer{}};
  }
  else if constexpr (_Size == 32 && ::cuda::std::is_same_v<_ParentT, ::cooperative_groups::thread_block>)
  {
    return this_warp{__hier_like};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_ParentT>,
                  "make_cg_equivalent_group has not been implemented for this thread_block_tile CG group yet");
  }
}
} // namespace cuda::experimental

#  endif // !_CCCL_DOXYGEN_INVOKED

#  include <cuda/std/__cccl/epilogue.h>

#endif // __has_include(<cooperative_groups.h>) && !_CUDAX_DISABLE_CG_INTEROP

#endif // _CUDA_EXPERIMENTAL___GROUP_CG_INTEROP_CUH
