//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/hierarchy>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/__group/queries.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental
{
template <::cuda::std::size_t... _UnitCounts>
struct __group_as_static_tag;

template <::cuda::std::size_t... _UnitCounts, bool _IsExhaustive>
class group_as<__group_as_static_tag<_UnitCounts...>, _IsExhaustive>
{
  static_assert(((_UnitCounts != 0) && ...), "all _UnitCounts must not be zero");
  static_assert((::cuda::std::in_range<unsigned>(_UnitCounts) && ...), "all _UnitCounts must be within uint32_t range");

  static constexpr auto __counts_sum = (0 + ... + _UnitCounts);

public:
  _CCCL_HIDE_FROM_ABI explicit group_as() = default;

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES(_IsExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(
    const ::cuda::std::integer_sequence<::cuda::std::size_t, _UnitCounts...>&) noexcept
  {}

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(
    const ::cuda::std::integer_sequence<::cuda::std::size_t, _UnitCounts...>&, const non_exhaustive_t&) noexcept
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return sizeof...(_UnitCounts);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= sizeof...(_UnitCounts))
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    constexpr ::cuda::std::size_t __counts[]{_UnitCounts...};
    return __counts[__i];
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned unit_count(::cuda::std::size_t __i) const noexcept
  {
    return static_cast<unsigned>(static_unit_count(__i));
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups = sizeof...(_UnitCounts);
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __prev_nunits      = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();
    constexpr auto __curr_ngroups = static_cast<unsigned>(sizeof...(_UnitCounts));
    const auto __ngroups          = __prev_mapping_result.group_count() * __curr_ngroups;

    if constexpr (_IsExhaustive)
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits == __counts_sum, "group_as mapping _IsExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits == static_cast<unsigned>(__counts_sum),
                     "group_as mapping _IsExhaustive precondition violation");
      }
    }
    else
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits >= __counts_sum, "group_as mapping requires more units than are available");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits >= static_cast<unsigned>(__counts_sum),
                     "group_as mapping requires more units than are available");
      }

      if (__prev_unit_rank >= static_cast<unsigned>(__counts_sum))
      {
        return _MappingResult::invalid_with_group_count(__ngroups);
      }
    }

    unsigned __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned __i = 0; __i < __curr_ngroups; ++__i)
    {
      const auto __i_count = unit_count(__i);
      if (__prev_unit_rank < __sum + __i_count)
      {
        const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __i;
        const auto __n          = __i_count;
        const auto __rank       = __prev_unit_rank - __sum;
        const auto __lane_mask =
          (::cuda::std::is_same_v<_Unit, thread_level>)
            ? ::cuda::experimental::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
                __prev_mapping_result.lane_mask(), __n, __rank)
            : __prev_mapping_result.lane_mask();
        return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
      }
      __sum += __i_count;
    }
    _CCCL_UNREACHABLE();
  }
};

template <::cuda::std::size_t _GroupCount>
struct __group_as_dynamic_tag;

template <::cuda::std::size_t _GroupCount, bool _IsExhaustive>
class group_as<__group_as_dynamic_tag<_GroupCount>, _IsExhaustive>
{
  static_assert(_GroupCount != ::cuda::std::dynamic_extent, "group_as requires static number of groups");

  unsigned __counts_[_GroupCount];

public:
  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES(_IsExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(::cuda::std::span<const unsigned, _GroupCount> __counts) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _GroupCount; ++__i)
    {
      _CCCL_ASSERT(__counts[__i] > 0, "none of the __counts can be 0");
      __counts_[__i] = __counts[__i];
    }
  }

  _CCCL_TEMPLATE(bool _IsExhaustive2 = _IsExhaustive)
  _CCCL_REQUIRES((!_IsExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(::cuda::std::span<const unsigned, _GroupCount> __counts,
                                               const non_exhaustive_t&) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _GroupCount; ++__i)
    {
      _CCCL_ASSERT(__counts[__i] > 0, "none of the __counts can be 0");
      __counts_[__i] = __counts[__i];
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return _GroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= _GroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned unit_count(::cuda::std::size_t __i) const noexcept
  {
    if (__i >= _GroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return __counts_[__i];
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups = _GroupCount;
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __prev_nunits      = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();
    constexpr auto __curr_ngroups = static_cast<unsigned>(_GroupCount);
    const auto __ngroups          = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsExhaustive)
    {
      _CCCL_ASSERT(::cuda::std::accumulate(__counts_, __counts_ + __curr_ngroups, 0u) == __prev_nunits,
                   "group_as mapping _IsExhaustive precondition violation");
    }
    else if (__prev_unit_rank >= ::cuda::std::accumulate(__counts_, __counts_ + __curr_ngroups, 0u))
    {
      return _MappingResult::invalid_with_group_count(__ngroups);
    }

    unsigned __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (unsigned __i = 0; __i < __curr_ngroups; ++__i)
    {
      const auto __i_count = unit_count(__i);
      if (__prev_unit_rank < __sum + __i_count)
      {
        const auto __group_rank = __prev_mapping_result.group_rank() * __curr_ngroups + __i;
        const auto __n          = __i_count;
        const auto __rank       = __prev_unit_rank - __sum;
        const auto __lane_mask =
          (::cuda::std::is_same_v<_Unit, thread_level>)
            ? ::cuda::experimental::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
                __prev_mapping_result.lane_mask(), __n, __rank)
            : __prev_mapping_result.lane_mask();
        return _MappingResult{__ngroups, __group_rank, __n, __rank, __lane_mask};
      }
      __sum += __i_count;
    }
    _CCCL_UNREACHABLE();
  }
};

template <::cuda::std::size_t... _UnitCounts>
_CCCL_DEVICE group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _UnitCounts...>&)
  -> group_as<__group_as_static_tag<_UnitCounts...>, true>;

template <::cuda::std::size_t... _UnitCounts>
_CCCL_DEVICE group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _UnitCounts...>&,
                      const non_exhaustive_t&) -> group_as<__group_as_static_tag<_UnitCounts...>, false>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp> _CCCL_AND ::cuda::std::
                 is_same_v<unsigned, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEVICE group_as(_Tp& __v) -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, true>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_spannable<_Tp> _CCCL_AND ::cuda::std::
                 is_same_v<unsigned, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEVICE group_as(_Tp& __v, const non_exhaustive_t&)
  -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, false>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_GROUP_AS_CUH
