//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bit_trunc.h>
#include <cuda/hierarchy>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/integral_constant_like.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/queries.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
// Previous mappings may split units into groups of different unit counts. However, the current mapping result only
// knows how many units are within the current group. That means that we need to do a collective operation to determine
// the group count and rank.
//
// This occurs only when group_by is used in composite_mapping as the second (or later) mapping, so it's not a common
// case. We currently implement the collective for threads within a warp groups.
template <class _PrevMappingResult>
inline constexpr bool __group_by_mapping_result_needs_fix_v =
  _PrevMappingResult::static_unit_count() == ::cuda::std::dynamic_extent
  && _PrevMappingResult::static_group_count() != 1;

template <class _Unit, class _ParentGroup, class _MappingResult>
[[nodiscard]] _CCCL_DEVICE_API _MappingResult
__group_by_fix_group_count_rank(const _Unit&, const _ParentGroup& __parent, const _MappingResult& __mapping_result)
{
  using ::cuda::std::is_same_v;
  using ::cuda::std::uint32_t;

  if constexpr (is_same_v<_Unit, thread_level> && is_same_v<typename _ParentGroup::level_type, warp_level>)
  {
    // Excluded threads must also participate in the parent-wide ballot.
    const auto __group_leaders = ::__ballot_sync(__parent.__mapping_result().lane_mask().value(),
                                                 __mapping_result.is_valid() && __mapping_result.unit_rank() == 0);

    if (!__mapping_result.is_valid())
    {
      return __mapping_result;
    }

    const auto __lane_mask  = __mapping_result.lane_mask();
    const auto __first_lane = ::cuda::std::countr_zero(__lane_mask.value());

    const auto __ngroups = static_cast<uint32_t>(::cuda::std::popcount(__group_leaders));
    const auto __group_rank =
      static_cast<uint32_t>(::cuda::std::popcount(::cuda::bit_trunc(__group_leaders, __first_lane)));
    return _MappingResult{
      __ngroups, __group_rank, __mapping_result.unit_count(), __mapping_result.unit_rank(), __lane_mask};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Unit>, "Not implemented");
  }
}

template <::cuda::std::size_t _StaticUnitCount>
class group_by
{
  static_assert(_StaticUnitCount != 0, "_StaticUnitCount must not be zero");
  static_assert(::cuda::std::in_range<::cuda::std::uint32_t>(_StaticUnitCount),
                "_StaticUnitCount must be within uint32_t range");

public:
  _CCCL_DEVICE_API explicit constexpr group_by(unsigned __unit_count)
  {
    _CCCL_ASSERT(__unit_count == _StaticUnitCount, "__unit_count must match the static _StaticUnitCount");
  }

  template <bool _Exhaustive, class _Unit, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static auto __do_map(const _Unit&, const _PrevMappingResult& __prev_mapping_result)
  {
    constexpr auto __unit_count = static_cast<::cuda::std::uint32_t>(_StaticUnitCount);

    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups =
      (__static_prev_nunits != ::cuda::std::dynamic_extent)
        ? __static_prev_nunits / _StaticUnitCount
        : ::cuda::std::dynamic_extent;
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent && __static_curr_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;
    constexpr bool __curr_is_always_exhaustive =
      __static_prev_nunits != ::cuda::std::dynamic_extent && __static_prev_nunits % _StaticUnitCount == 0;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       _StaticUnitCount,
                       _Exhaustive || (_PrevMappingResult::is_always_exhaustive() && __curr_is_always_exhaustive),
                       _PrevMappingResult::is_always_contiguous()>;

    if constexpr (!_PrevMappingResult::is_always_exhaustive())
    {
      if (!__prev_mapping_result.is_valid())
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __prev_nunits     = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank  = __prev_mapping_result.unit_rank();
    const auto __curr_ngroups    = __prev_nunits / __unit_count;
    const auto __curr_group_rank = __prev_unit_rank / __unit_count;
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise return invalid mapping for the remainder.
    if constexpr (_Exhaustive)
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits % _StaticUnitCount == 0,
                      "group_by mapping would produce units without a group during exhaustive mapping");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits % __unit_count == 0,
                     "group_by mapping would produce units without a group during exhaustive mapping");
      }
    }
    else if constexpr (!__curr_is_always_exhaustive)
    {
      if (__prev_nunits % __unit_count != 0 && __curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank;
    const auto __n          = __unit_count;
    const auto __rank       = __prev_unit_rank % __n;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __n, __rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const
  {
    const auto __ret = __do_map<true>(__unit, __prev_mapping_result);
    if constexpr (__group_by_mapping_result_needs_fix_v<_PrevMappingResult>)
    {
      return ::cuda::experimental::coop::__group_by_fix_group_count_rank(__unit, __parent, __ret);
    }
    else
    {
      return __ret;
    }
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup& __parent,
      const _PrevMappingResult& __prev_mapping_result) const
  {
    const auto __ret = __do_map<false>(__unit, __prev_mapping_result);
    if constexpr (__group_by_mapping_result_needs_fix_v<_PrevMappingResult>)
    {
      return ::cuda::experimental::coop::__group_by_fix_group_count_rank(__unit, __parent, __ret);
    }
    else
    {
      return __ret;
    }
  }
};

template <>
class group_by<::cuda::std::dynamic_extent>
{
  ::cuda::std::uint32_t __unit_count_;

public:
  _CCCL_DEVICE_API explicit constexpr group_by(::cuda::std::uint32_t __unit_count)
      : __unit_count_{__unit_count}
  {
    _CCCL_ASSERT(__unit_count > 0, "__unit_count must be greater than 0");
  }

  template <bool _Exhaustive, class _Unit, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto __do_map(const _Unit&, const _PrevMappingResult& __prev_mapping_result) const
  {
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

    const auto __prev_nunits     = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank  = __prev_mapping_result.unit_rank();
    const auto __curr_ngroups    = __prev_nunits / __unit_count_;
    const auto __curr_group_rank = __prev_unit_rank / __unit_count_;
    const auto __ngroups         = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_Exhaustive)
    {
      _CCCL_ASSERT(__prev_nunits % __unit_count_ == 0,
                   "group_by mapping would produce units without a group during exhaustive mapping");
    }
    else if (__prev_nunits % __unit_count_ != 0)
    {
      if (__curr_group_rank >= __curr_ngroups)
      {
        return _MappingResult::__invalid();
      }
    }

    const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __curr_group_rank;
    const auto __n          = __unit_count_;
    const auto __rank       = __prev_unit_rank % __unit_count_;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __n, __rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const
  {
    const auto __ret = __do_map<true>(__unit, __prev_mapping_result);
    if constexpr (__group_by_mapping_result_needs_fix_v<_PrevMappingResult>)
    {
      return ::cuda::experimental::coop::__group_by_fix_group_count_rank(__unit, __parent, __ret);
    }
    else
    {
      return __ret;
    }
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup& __parent,
      const _PrevMappingResult& __prev_mapping_result) const
  {
    const auto __ret = __do_map<false>(__unit, __prev_mapping_result);
    if constexpr (__group_by_mapping_result_needs_fix_v<_PrevMappingResult>)
    {
      return ::cuda::experimental::coop::__group_by_fix_group_count_rank(__unit, __parent, __ret);
    }
    else
    {
      return __ret;
    }
  }
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_by(_Tp) -> group_by<::cuda::std::__maybe_static_ext<_Tp>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_BY_CUH
