//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_THIS_GROUP_CUH
#define _CUDA_EXPERIMENTAL___GROUP_THIS_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__warp/lane_mask.h>
#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/implicit_hierarchy.cuh>
#include <cuda/experimental/__group/synchronizer/level_synchronizer.cuh>

#if _CCCL_HAS_COOPERATIVE_GROUPS()
#  include <cooperative_groups.h>
#endif // _CCCL_HAS_COOPERATIVE_GROUPS()

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _HierarchyLike>
using __hierarchy_type_of =
  ::cuda::std::remove_cvref_t<decltype(::cuda::__unpack_hierarchy_if_needed(::cuda::std::declval<_HierarchyLike>()))>;

template <class _Level>
struct __this_mapping_result
{
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    return 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count() noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned unit_count() const noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned unit_rank() const noexcept
  {
    return 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const noexcept
  {
    if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
    {
      return ::cuda::device::lane_mask::this_lane();
    }
    else
    {
      return ::cuda::device::lane_mask::all();
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
  {
    return true;
  }
};

template <class _Level, class _Hierarchy>
class __this_group_base
{
  static_assert(__is_hierarchy_level_v<_Level>);
  static_assert(__is_hierarchy_v<_Hierarchy>);

  using _SynchronizerInstance = level_synchronizer::__synchronizer_instance<_Level>;

public:
  using unit_type             = _Level;
  using level_type            = _Level;
  using mapping_type          = void;
  using __mapping_result_type = __this_mapping_result<_Level>;
  using hierarchy_type        = _Hierarchy;
  using synchronizer_type     = level_synchronizer;

private:
  _Hierarchy __hier_;
  __mapping_result_type __mapping_result_{};
  level_synchronizer __synchronizer_{};
  _SynchronizerInstance __synchronizer_instance_{};

public:
  _CCCL_DEVICE_API explicit __this_group_base() noexcept
      : __hier_{::cuda::experimental::__implicit_hierarchy()}
  {}

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API __this_group_base(const _HierarchyLike& __hier_like) noexcept
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  [[nodiscard]] _CCCL_DEVICE_API const hierarchy_type& hierarchy() const noexcept
  {
    return __hier_;
  }

  // this groups don't have a mapping type
  auto mapping() const = delete;

  // todo(dabayer): Do we want to expose mapping result getter?
  [[nodiscard]] _CCCL_DEVICE_API const __mapping_result_type& __mapping_result() const noexcept
  {
    return __mapping_result_;
  }

  [[nodiscard]] _CCCL_DEVICE_API const synchronizer_type& synchronizer() const noexcept
  {
    return __synchronizer_;
  }

  _CCCL_DEVICE_API void sync() const noexcept
  {
    __synchronizer_instance_.do_sync(__mapping_result_, __synchronizer_, __hier_);
  }

  _CCCL_DEVICE_API void sync_aligned() const noexcept
  {
    __synchronizer_instance_.do_sync_aligned(__mapping_result_, __synchronizer_, __hier_);
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template count_as<_Tp>(__in_level, __hier_);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.count(__in_level, __hier_);
  }

#  if _CCCL_CUDA_COMPILATION()
  _CCCL_TEMPLATE(class _Tp, class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel> _CCCL_AND(
    !::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.template rank_as<_Tp>(__in_level, __hier_);
  }

  _CCCL_TEMPLATE(class _InLevel, class _Level2 = _Level)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel> _CCCL_AND(!::cuda::std::is_same_v<_Level2, grid_level>))
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const noexcept
  {
    return _Level{}.rank(__in_level, __hier_);
  }
#  endif // _CCCL_CUDA_COMPILATION()
};

template <class _Hierarchy>
class this_thread : public __this_group_base<thread_level, _Hierarchy>
{
  using __base_type = __this_group_base<thread_level, _Hierarchy>;

public:
  using __base_type::__base_type;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  template <class _Parent>
  _CCCL_DEVICE_API this_thread(const ::cooperative_groups::thread_block_tile<1, _Parent>&) noexcept
  {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_thread() -> this_thread<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_thread(const _Hierarchy&) -> this_thread<__hierarchy_type_of<_Hierarchy>>;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_thread(const ::cooperative_groups::thread_block_tile<1, void>&)
  -> this_thread<__implicit_hierarchy_t>;
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_warp : public __this_group_base<warp_level, _Hierarchy>
{
  using __base_type = __this_group_base<warp_level, _Hierarchy>;

public:
  using __base_type::__base_type;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  template <class _Parent>
  _CCCL_DEVICE_API this_warp(const ::cooperative_groups::thread_block_tile<32, _Parent>&) noexcept
  {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_warp() -> this_warp<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_warp(const _Hierarchy&) -> this_warp<__hierarchy_type_of<_Hierarchy>>;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
template <class _Parent>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_warp(const ::cooperative_groups::thread_block_tile<32, _Parent>&)
  -> this_warp<__implicit_hierarchy_t>;
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_block : public __this_group_base<block_level, _Hierarchy>
{
  using __base_type = __this_group_base<block_level, _Hierarchy>;

public:
  using __base_type::__base_type;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  _CCCL_DEVICE_API this_block(const ::cooperative_groups::thread_block&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_block() -> this_block<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_block(const _Hierarchy&) -> this_block<__hierarchy_type_of<_Hierarchy>>;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_block(const ::cooperative_groups::thread_block&)
  -> this_block<__implicit_hierarchy_t>;
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

template <class _Hierarchy>
class this_cluster : public __this_group_base<cluster_level, _Hierarchy>
{
  using __base_type = __this_group_base<cluster_level, _Hierarchy>;

public:
  using __base_type::__base_type;

#  if _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
  _CCCL_DEVICE_API this_cluster(const ::cooperative_groups::cluster_group&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_cluster() -> this_cluster<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_cluster(const _Hierarchy&) -> this_cluster<__hierarchy_type_of<_Hierarchy>>;

#  if _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_cluster(const ::cooperative_groups::cluster_group&)
  -> this_cluster<__implicit_hierarchy_t>;
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS() && defined(_CG_HAS_CLUSTER_GROUP)

template <class _Hierarchy>
class this_grid : public __this_group_base<grid_level, _Hierarchy>
{
  using __base_type = __this_group_base<grid_level, _Hierarchy>;

public:
  using __base_type::__base_type;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
  _CCCL_DEVICE_API this_grid(const ::cooperative_groups::grid_group&) noexcept {}
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_grid() -> this_grid<__implicit_hierarchy_t>;

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_grid(const _Hierarchy&) -> this_grid<__hierarchy_type_of<_Hierarchy>>;

#  if _CCCL_HAS_COOPERATIVE_GROUPS()
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_grid(const ::cooperative_groups::grid_group&)
  -> this_grid<__implicit_hierarchy_t>;
#  endif // _CCCL_HAS_COOPERATIVE_GROUPS()

_CCCL_TEMPLATE(class _Level, class... _Args)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Level>)
[[nodiscard]] _CCCL_DEVICE_API auto make_this_group(const _Level&, _Args&&... __args) noexcept
{
  if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
  {
    return this_thread{::cuda::std::forward<_Args>(__args)...};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, warp_level>)
  {
    return this_warp{::cuda::std::forward<_Args>(__args)...};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, block_level>)
  {
    return this_block{::cuda::std::forward<_Args>(__args)...};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
  {
    return this_cluster{::cuda::std::forward<_Args>(__args)...};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, grid_level>)
  {
    return this_grid{::cuda::std::forward<_Args>(__args)...};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Level>, "unknown _Level");
  }
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_THIS_GROUP_CUH
