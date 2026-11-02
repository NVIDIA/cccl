//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH

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
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _RangeView>
class group_as
{
  static_assert(::cuda::std::ranges::sized_range<_RangeView>, "_RangeView must be a sized range");
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_RangeView>, ::cuda::std::uint32_t>,
                "_RangeView's value_type must be convertible to uint32_t");

  _RangeView __unit_counts_;

public:
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API group_as(_Range&& __unit_counts) noexcept(
    noexcept(::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__unit_counts))))
      : __unit_counts_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__unit_counts))}
  {}

  template <bool _Exhaustive, class _Unit, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  __do_map(const _Unit&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    // todo(dabayer): Make group count potentially statically-known when _RangeView has known a known static size.
    using _MappingResult =
      __mapping_result<::cuda::std::dynamic_extent,
                       ::cuda::std::dynamic_extent,
                       _Exhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if constexpr (!_PrevMappingResult::is_always_exhaustive())
    {
      if (!__prev_mapping_result.is_valid())
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __prev_nunits    = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank = __prev_mapping_result.unit_rank();
    const auto __curr_ngroups   = static_cast<::cuda::std::uint32_t>(::cuda::std::ranges::size(__unit_counts_));
    const auto __ngroups        = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_Exhaustive)
    {
      _CCCL_ASSERT(::cuda::std::accumulate(
                     ::cuda::std::ranges::cbegin(__unit_counts_), ::cuda::std::ranges::cend(__unit_counts_), 0u)
                     == __prev_nunits,
                   "group_as mapping would produce units without a group during exhaustive mapping");
    }
    else
    {
      _CCCL_ASSERT(::cuda::std::accumulate(
                     ::cuda::std::ranges::cbegin(__unit_counts_), ::cuda::std::ranges::cend(__unit_counts_), 0u)
                     <= __prev_nunits,
                   "group_as mapping requires more units than are available");

      if (__prev_unit_rank >= ::cuda::std::accumulate(
            ::cuda::std::ranges::cbegin(__unit_counts_), ::cuda::std::ranges::cend(__unit_counts_), 0u))
      {
        return _MappingResult::__invalid();
      }
    }

    ::cuda::std::uint32_t __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::uint32_t __i = 0; __i < __curr_ngroups; ++__i)
    {
      const auto __i_count = static_cast<::cuda::std::uint32_t>(__unit_counts_[__i]);
      if (__prev_unit_rank < __sum + __i_count)
      {
        const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __i;
        const auto __n          = __i_count;
        const auto __rank       = __prev_unit_rank - __sum;
        const auto __lane_mask =
          (::cuda::std::is_same_v<_Unit, thread_level>)
            ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
                __prev_mapping_result.lane_mask(), __n, __rank)
            : __prev_mapping_result.lane_mask();
        return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
      }
      __sum += __i_count;
    }
    _CCCL_UNREACHABLE();
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    return __do_map<true>(__unit, __prev_mapping_result);
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup&,
      const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    return __do_map<false>(__unit, __prev_mapping_result);
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<_Range> _CCCL_AND ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, ::cuda::std::uint32_t>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(_Range&&) -> group_as<::cuda::std::ranges::views::all_t<_Range>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH
