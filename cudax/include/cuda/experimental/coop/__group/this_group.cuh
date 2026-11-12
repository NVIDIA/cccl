//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_THIS_GROUP_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_THIS_GROUP_CUH

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
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/group_view.cuh>
#include <cuda/experimental/coop/__group/synchronizer/level_synchronizer.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Level>
struct __this_mapping_result
{
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_rank() const
  {
    return 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    _CCCL_ASSERT(__i < static_group_count(), "__i out of range");
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    return 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
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

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return true;
  }
};

template <class _Level, class _Hierarchy>
class __this_group_base
{
  static_assert(__is_hierarchy_level_v<_Level>);
  static_assert(__is_hierarchy_v<_Hierarchy>);

  using _MappingResult _CCCL_NODEBUG        = __this_mapping_result<_Level>;
  using _SynchronizerInstance _CCCL_NODEBUG = level_synchronizer::__synchronizer_instance<_Level>;

  _Hierarchy __hier_;
  _MappingResult __mapping_result_{};
  _SynchronizerInstance __synchronizer_instance_{};

public:
  using unit_type                    = _Level;
  using level_type                   = _Level;
  using __mapping_result_type        = _MappingResult;
  using __synchronizer_instance_type = _SynchronizerInstance;
  using hierarchy_type               = _Hierarchy;
  using arrival_token                = typename _SynchronizerInstance::arrival_token;

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API __this_group_base(const _HierarchyLike& __hier_like)
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  // Groups can't be copied, moved nor assigned.
  __this_group_base(const __this_group_base&)            = delete;
  __this_group_base(__this_group_base&&)                 = delete;
  __this_group_base& operator=(const __this_group_base&) = delete;
  __this_group_base& operator=(__this_group_base&&)      = delete;

  //! @brief Invokes the callable with a view specialized for this group's static unit count.
  //! Counts that are only known at runtime remain dynamic.
  template <class _Fn>
  _CCCL_DEVICE_API decltype(auto) with_static_unit_count(_Fn&& __fn) const
  {
    return ::cuda::experimental::coop::__with_static_unit_count(*this, ::cuda::std::forward<_Fn>(__fn));
  }

  [[nodiscard]] _CCCL_DEVICE_API const hierarchy_type& hierarchy() const
  {
    return __hier_;
  }

  // todo(dabayer): Do we want to expose mapping result getter?
  [[nodiscard]] _CCCL_DEVICE_API const __mapping_result_type& __mapping_result() const
  {
    return __mapping_result_;
  }

  [[nodiscard]] _CCCL_DEVICE_API const _SynchronizerInstance& __synchronizer_instance() const
  {
    return __synchronizer_instance_;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return __mapping_result_type::is_always_contiguous();
  }

  _CCCL_DEVICE_API void sync() const
  {
    __synchronizer_instance_.do_sync(__hier_);
  }

  _CCCL_DEVICE_API void sync_aligned() const
  {
    __synchronizer_instance_.do_sync_aligned(__hier_);
  }

  [[nodiscard]] _CCCL_DEVICE_API arrival_token arrive() const
  {
    return __synchronizer_instance_.do_arrive(__hier_);
  }

  [[nodiscard]] _CCCL_DEVICE_API arrival_token arrive_aligned() const
  {
    return __synchronizer_instance_.do_arrive_aligned(__hier_);
  }

  _CCCL_DEVICE_API void wait(arrival_token&& __tok) const
  {
    __synchronizer_instance_.do_wait(__hier_, ::cuda::std::move(__tok));
  }

  _CCCL_DEVICE_API void wait_aligned(arrival_token&& __tok) const
  {
    __synchronizer_instance_.do_wait_aligned(__hier_, ::cuda::std::move(__tok));
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(const _InLevel& __in_level)
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, _Level>)
    {
      return 1;
    }
    else
    {
      return ::cuda::__static_count_query<_Level, _InLevel, _Hierarchy>();
    }
  }

  template <class _Tp, class _MappingResult, class _InLevel>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __count_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level)
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, _Level>)
    {
      return _Tp{1};
    }
    else
    {
      return __count_query<_Level, _InLevel>::template __call<_Tp>(__hier);
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _InLevel& __in_level) const
  {
    return __count_as_impl<_Tp>(__mapping_result_, __hier_, __in_level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _InLevel& __in_level) const
  {
    return __count_as_impl<typename _InLevel::__product_type>(__mapping_result_, __hier_, __in_level);
  }

  template <class _Tp, class _MappingResult, class _InLevel>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp
  __rank_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level)
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, _Level>)
    {
      return _Tp{0};
    }
    else
    {
      return __rank_query<_Level, _InLevel>::template __call<_Tp>(__hier);
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const
  {
    return __rank_as_impl<_Tp>(__mapping_result_, __hier_, __in_level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const
  {
    return __rank_as_impl<typename _InLevel::__product_type>(__mapping_result_, __hier_, __in_level);
  }
};

template <class _Hierarchy>
class this_thread : public __this_group_base<thread_level, _Hierarchy>
{
  using __base_type = __this_group_base<thread_level, _Hierarchy>;

public:
  using __base_type::__base_type;
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_thread(const _Hierarchy&) -> this_thread<__hierarchy_type_of<_Hierarchy>>;

template <class _Hierarchy>
class this_warp : public __this_group_base<warp_level, _Hierarchy>
{
  using __base_type = __this_group_base<warp_level, _Hierarchy>;

public:
  using __base_type::__base_type;
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_warp(const _Hierarchy&) -> this_warp<__hierarchy_type_of<_Hierarchy>>;

template <class _Hierarchy>
class this_block : public __this_group_base<block_level, _Hierarchy>
{
  using __base_type = __this_group_base<block_level, _Hierarchy>;

public:
  using __base_type::__base_type;
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_block(const _Hierarchy&) -> this_block<__hierarchy_type_of<_Hierarchy>>;

template <class _Hierarchy>
class this_cluster : public __this_group_base<cluster_level, _Hierarchy>
{
  using __base_type = __this_group_base<cluster_level, _Hierarchy>;

public:
  using __base_type::__base_type;
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_cluster(const _Hierarchy&) -> this_cluster<__hierarchy_type_of<_Hierarchy>>;

template <class _Hierarchy>
class this_grid : public __this_group_base<grid_level, _Hierarchy>
{
  using __base_type = __this_group_base<grid_level, _Hierarchy>;

public:
  using __base_type::__base_type;
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES this_grid(const _Hierarchy&) -> this_grid<__hierarchy_type_of<_Hierarchy>>;

_CCCL_TEMPLATE(class _Level, class _HierarchyLike)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Level> _CCCL_AND __is_or_has_hierarchy_member_v<_HierarchyLike>)
[[nodiscard]] _CCCL_DEVICE_API auto make_this_group(const _Level&, const _HierarchyLike& __hier_like)
{
  if constexpr (::cuda::std::is_same_v<_Level, thread_level>)
  {
    return this_thread{__hier_like};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, warp_level>)
  {
    return this_warp{__hier_like};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, block_level>)
  {
    return this_block{__hier_like};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, cluster_level>)
  {
    return this_cluster{__hier_like};
  }
  else if constexpr (::cuda::std::is_same_v<_Level, grid_level>)
  {
    return this_grid{__hier_like};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Level>, "unknown _Level");
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_THIS_GROUP_CUH
