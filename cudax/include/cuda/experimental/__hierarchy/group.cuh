//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH
#define _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH

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
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__hierarchy/fwd.cuh>
#include <cuda/experimental/__hierarchy/grid_sync.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _HierarchyLike>
using __hierarchy_type_of =
  ::cuda::std::remove_cvref_t<decltype(::cuda::__unpack_hierarchy_if_needed(::cuda::std::declval<_HierarchyLike>()))>;

// todo: use __hier_ in queries
template <class _Level, class _Hierarchy>
class __hierarchy_group_base<_Level, _Hierarchy, __this_hierarchy_group_kind>
{
  static_assert(__is_hierarchy_level_v<_Level>);
  static_assert(__is_hierarchy_v<_Hierarchy>);

  const _Hierarchy& __hier_;

public:
  using hierarchy_type = _Hierarchy;

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API __hierarchy_group_base(const _HierarchyLike& __hier_like) noexcept
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  [[nodiscard]] _CCCL_API const _Hierarchy& hierarchy() const noexcept
  {
    return __hier_;
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_API constexpr _Tp count_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template count_as<_Tp>(__in_level);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_API constexpr auto count(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.count(__in_level);
  }

#if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template rank_as<_Tp>(__in_level);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.rank(__in_level);
  }
#endif // _CCCL_CUDA_COMPILATION()
};

template <class _Hierarchy>
class thread_group<_Hierarchy, __this_hierarchy_group_kind> : __this_hierarchy_group_base<thread_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<thread_level, _Hierarchy>;

public:
  using level_type = thread_level;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::hierarchy;
#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

  _CCCL_DEVICE_API void sync() noexcept {}
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE thread_group(const _Hierarchy&)
  -> thread_group<__hierarchy_type_of<_Hierarchy>, __this_hierarchy_group_kind>;

template <class _Hierarchy>
class warp_group<_Hierarchy, __this_hierarchy_group_kind> : __this_hierarchy_group_base<warp_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<warp_level, _Hierarchy>;

public:
  using level_type = warp_level;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::hierarchy;
#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncwarp();
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE warp_group(const _Hierarchy&)
  -> warp_group<__hierarchy_type_of<_Hierarchy>, __this_hierarchy_group_kind>;

template <class _Hierarchy>
class block_group<_Hierarchy, __this_hierarchy_group_kind> : __this_hierarchy_group_base<block_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<block_level, _Hierarchy>;

public:
  using level_type = block_level;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::hierarchy;

#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

  _CCCL_DEVICE_API void sync() noexcept
  {
    ::__syncthreads();
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE block_group(const _Hierarchy&)
  -> block_group<__hierarchy_type_of<_Hierarchy>, __this_hierarchy_group_kind>;

template <class _Hierarchy>
class cluster_group<_Hierarchy, __this_hierarchy_group_kind> : __this_hierarchy_group_base<cluster_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<cluster_level, _Hierarchy>;

public:
  using level_type = cluster_level;

  using __base_type::__base_type;
  using __base_type::count;
  using __base_type::count_as;
  using __base_type::hierarchy;

#if _CCCL_CUDA_COMPILATION()
  using __base_type::rank;
  using __base_type::rank_as;

  _CCCL_DEVICE_API void sync() noexcept
  {
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      ({
                        ::__cluster_barrier_arrive();
                        ::__cluster_barrier_wait();
                      }),
                      ({ ::__syncthreads(); }))
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE cluster_group(const _Hierarchy&)
  -> cluster_group<__hierarchy_type_of<_Hierarchy>, __this_hierarchy_group_kind>;

template <class _Hierarchy>
class grid_group<_Hierarchy, __this_hierarchy_group_kind> : __this_hierarchy_group_base<grid_level, _Hierarchy>
{
  using __base_type = __this_hierarchy_group_base<grid_level, _Hierarchy>;

public:
  using level_type = grid_level;

  using __base_type::__base_type;
  using __base_type::hierarchy;

#if _CCCL_CUDA_COMPILATION()
  _CCCL_DEVICE_API void sync() noexcept
  {
    ::cuda::experimental::__cg_imported::__grid_sync();
  }
#endif // _CCCL_CUDA_COMPILATION()
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_HOST_DEVICE grid_group(const _Hierarchy&)
  -> grid_group<__hierarchy_type_of<_Hierarchy>, __this_hierarchy_group_kind>;

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_thread(const _HierarchyLike& __hier_like) noexcept
{
  return thread_group{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_warp(const _HierarchyLike& __hier_like) noexcept
{
  return warp_group{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_block(const _HierarchyLike& __hier_like) noexcept
{
  return block_group{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_cluster(const _HierarchyLike& __hier_like) noexcept
{
  return cluster_group{__hier_like};
}

_CCCL_TEMPLATE(class _HierarchyLike)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto this_grid(const _HierarchyLike& __hier_like) noexcept
{
  return grid_group{__hier_like};
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___HIERARCHY_GROUP_CUH
