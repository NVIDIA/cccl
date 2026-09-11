//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_COALESCED_GROUP_CUH
#define _CUDA_EXPERIMENTAL___GROUP_COALESCED_GROUP_CUH

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
#include <cuda/std/cstdint>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/__group/synchronizer/lane_synchronizer.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
class __coalesced_mapping_result
{
  ::cuda::device::lane_mask __lane_mask_;
  ::cuda::std::uint32_t __unit_count_;
  ::cuda::std::uint32_t __unit_rank_;

public:
  _CCCL_DEVICE_API __coalesced_mapping_result() noexcept
      : __lane_mask_{::cuda::device::lane_mask::all_active()}
      , __unit_count_{static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__lane_mask_.value()))}
      , __unit_rank_{static_cast<::cuda::std::uint32_t>(
          ::cuda::std::popcount((__lane_mask_ & ::cuda::device::lane_mask::all_less()).value()))}
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const noexcept
  {
    return 1;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_rank() const noexcept
  {
    return 0;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count() noexcept
  {
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const noexcept
  {
    return __unit_count_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const noexcept
  {
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const noexcept
  {
    return __lane_mask_;
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
  _SynchronizerInstance __synchronizer_instance_{};

public:
  using unit_type             = thread_level;
  using level_type            = warp_level;
  using hierarchy_type        = _Hierarchy;
  using __mapping_result_type = _MappingResult;

  _CCCL_TEMPLATE(class _HierarchyLike)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Hierarchy, __hierarchy_type_of<_HierarchyLike>>)
  _CCCL_DEVICE_API explicit coalesced_group(const _HierarchyLike& __hier_like) noexcept
      : __hier_{::cuda::__unpack_hierarchy_if_needed(__hier_like)}
  {}

  // Groups can't be copied, moved nor assigned.
  coalesced_group(const coalesced_group&)            = delete;
  coalesced_group(coalesced_group&&)                 = delete;
  coalesced_group& operator=(const coalesced_group&) = delete;
  coalesced_group& operator=(coalesced_group&&)      = delete;

  [[nodiscard]] _CCCL_DEVICE_API const hierarchy_type& hierarchy() const noexcept
  {
    return __hier_;
  }

  [[nodiscard]] _CCCL_DEVICE_API _MappingResult __mapping_result() const noexcept
  {
    return __mapping_result_;
  }

  [[nodiscard]] _CCCL_DEVICE_API const _SynchronizerInstance& __synchronizer_instance() const noexcept
  {
    return __synchronizer_instance_;
  }

  // todo(dabayer): Do we want to expose .arrive() and .wait()? Do we want to implement .sync() using them? Do we want
  //                aligned/unaligned variants?
  _CCCL_DEVICE_API void sync() const noexcept
  {
    __synchronizer_instance_.do_sync(__mapping_result_, __hier_);
  }

  _CCCL_DEVICE_API void sync_aligned() const noexcept
  {
    __synchronizer_instance_.do_sync_aligned(__mapping_result_, __hier_);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(const _InLevel& __in_level) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return 1;
    }
    else
    {
      return level_type::static_count(__in_level, _Hierarchy{});
    }
  }

  template <class _Tp, class _MappingResult, class _InLevel>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __count_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return _Tp{1};
    }
    else
    {
      return level_type::template count_as<_Tp>(__in_level, __hier);
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _InLevel& __in_level) const noexcept
  {
    return __count_as_impl<_Tp>(__mapping_result_, __hier_, __in_level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _InLevel& __in_level) const noexcept
  {
    return __count_as_impl<typename _InLevel::__product_type>(__mapping_result_, __hier_, __in_level);
  }

  template <class _Tp, class _MappingResult, class _InLevel>
  [[nodiscard]] _CCCL_DEVICE_API static _Tp
  __rank_as_impl(const _MappingResult&, const _Hierarchy& __hier, const _InLevel& __in_level) noexcept
  {
    if constexpr (::cuda::std::is_same_v<_InLevel, level_type>)
    {
      return _Tp{0};
    }
    else
    {
      return level_type::template rank_as<_Tp>(__in_level, __hier);
    }
  }

  _CCCL_TEMPLATE(class _Tp, class _InLevel)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp> _CCCL_AND __is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _InLevel& __in_level) const noexcept
  {
    return __rank_as_impl<_Tp>(__mapping_result_, __hier_, __in_level);
  }

  _CCCL_TEMPLATE(class _InLevel)
  _CCCL_REQUIRES(__is_hierarchy_level_v<_InLevel>)
  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _InLevel& __in_level) const noexcept
  {
    return __rank_as_impl<typename _InLevel::__product_type>(__mapping_result_, __hier_, __in_level);
  }
};

_CCCL_TEMPLATE(class _Hierarchy)
_CCCL_REQUIRES(__is_or_has_hierarchy_member_v<_Hierarchy>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES coalesced_group(const _Hierarchy&) -> coalesced_group<__hierarchy_type_of<_Hierarchy>>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_COALESCED_GROUP_CUH
