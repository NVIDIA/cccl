//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_COALESCED_GROUP_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_COALESCED_GROUP_CUH

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
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/group_view.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/synchronizer/lane_synchronizer.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
class __coalesced_mapping_result
{
  ::cuda::device::lane_mask __lane_mask_;
  ::cuda::std::uint32_t __unit_count_;
  ::cuda::std::uint32_t __unit_rank_;

public:
  _CCCL_DEVICE_API __coalesced_mapping_result()
      : __lane_mask_{::cuda::device::lane_mask::all_active()}
      , __unit_count_{static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__lane_mask_.value()))}
      , __unit_rank_{static_cast<::cuda::std::uint32_t>(
          ::cuda::std::popcount((__lane_mask_ & ::cuda::device::lane_mask::all_less()).value()))}
  {}

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
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    _CCCL_ASSERT(__i < static_group_count(), "__i out of range");
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    return __unit_count_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    return __lane_mask_;
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
    return false;
  }
};

template <class _Hierarchy>
class coalesced_group
{
  using _MappingResult _CCCL_NODEBUG        = __coalesced_mapping_result;
  using _SynchronizerInstance _CCCL_NODEBUG = lane_synchronizer::__synchronizer_instance;

  _Hierarchy __hier_;
  _MappingResult __mapping_result_{};
  _SynchronizerInstance __synchronizer_instance_{__mapping_result_.lane_mask()};

public:
  using unit_type                    = thread_level;
  using level_type                   = warp_level;
  using hierarchy_type               = _Hierarchy;
  using __mapping_result_type        = _MappingResult;
  using __synchronizer_instance_type = _SynchronizerInstance;
  using arrival_token                = typename _SynchronizerInstance::arrival_token;

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API explicit coalesced_group(const _HierarchyLike& __hier_like)
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  // Groups can't be copied, moved nor assigned.
  coalesced_group(const coalesced_group&)            = delete;
  coalesced_group(coalesced_group&&)                 = delete;
  coalesced_group& operator=(const coalesced_group&) = delete;
  coalesced_group& operator=(coalesced_group&&)      = delete;

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

  [[nodiscard]] _CCCL_DEVICE_API _MappingResult __mapping_result() const
  {
    return __mapping_result_;
  }

  [[nodiscard]] _CCCL_DEVICE_API const _SynchronizerInstance& __synchronizer_instance() const
  {
    return __synchronizer_instance_;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _MappingResult::is_always_contiguous();
  }

  // todo(dabayer): Do we want to expose .arrive() and .wait()? Do we want to implement .sync() using them? Do we want
  //                aligned/unaligned variants?
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
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return 1;
    }
    else
    {
      return ::cuda::__static_count_query<level_type, _InLevel, _Hierarchy>();
    }
  }

  template <class _Tp, class _MappingResult, class _InLevel>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __count_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level)
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return _Tp{1};
    }
    else
    {
      return __count_query<level_type, _InLevel>::template __call<_Tp>(__hier);
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
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __rank_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level)
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return _Tp{0};
    }
    else
    {
      return __rank_query<level_type, _InLevel>::template __call<_Tp>(__hier);
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp rank_as(const _InLevel& __in_level) const
  {
    return __rank_as_impl<_Tp>(__mapping_result_, __hier_, __in_level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto rank(const _InLevel& __in_level) const
  {
    return __rank_as_impl<typename _InLevel::__product_type>(__mapping_result_, __hier_, __in_level);
  }
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES coalesced_group(const _Hierarchy&) -> coalesced_group<__hierarchy_type_of<_Hierarchy>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_COALESCED_GROUP_CUH
