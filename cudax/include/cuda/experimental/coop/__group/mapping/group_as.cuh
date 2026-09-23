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
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/queries.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental::coop
{
template <::cuda::std::size_t... _StaticUnitCounts>
struct __group_as_static_tag;

template <::cuda::std::size_t... _StaticUnitCounts, bool _IsAlwaysExhaustive>
class group_as<__group_as_static_tag<_StaticUnitCounts...>, _IsAlwaysExhaustive>
{
  static_assert(((_StaticUnitCounts != 0) && ...), "all _StaticUnitCounts must not be zero");
  static_assert((::cuda::std::in_range<::cuda::std::uint32_t>(_StaticUnitCounts) && ...),
                "all _StaticUnitCounts must be within uint32_t range");

  static constexpr auto __unit_counts_sum = (0 + ... + _StaticUnitCounts);

public:
  _CCCL_HIDE_FROM_ABI explicit group_as() = default;

  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES(_IsAlwaysExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(
    const ::cuda::std::integer_sequence<::cuda::std::size_t, _StaticUnitCounts...>&) noexcept
  {}

  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES((!_IsAlwaysExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(
    const ::cuda::std::integer_sequence<::cuda::std::size_t, _StaticUnitCounts...>&, const non_exhaustive_t&) noexcept
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return sizeof...(_StaticUnitCounts);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= sizeof...(_StaticUnitCounts))
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    constexpr ::cuda::std::size_t __counts[]{_StaticUnitCounts...};
    return __counts[__i];
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t unit_count(::cuda::std::size_t __i) const noexcept
  {
    return static_cast<::cuda::std::uint32_t>(static_unit_count(__i));
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr ::cuda::std::uint32_t __unit_counts[]{static_cast<::cuda::std::uint32_t>(_StaticUnitCounts)...};

    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups = sizeof...(_StaticUnitCounts);
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsAlwaysExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_nunits      = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();
    constexpr auto __curr_ngroups = static_cast<::cuda::std::uint32_t>(sizeof...(_StaticUnitCounts));
    const auto __ngroups          = __prev_mapping_result.group_count() * __curr_ngroups;

    if constexpr (_IsAlwaysExhaustive)
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits == __unit_counts_sum,
                      "group_as mapping _IsAlwaysExhaustive precondition violation");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits == static_cast<::cuda::std::uint32_t>(__unit_counts_sum),
                     "group_as mapping _IsAlwaysExhaustive precondition violation");
      }
    }
    else
    {
      if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
      {
        static_assert(__static_prev_nunits >= __unit_counts_sum,
                      "group_as mapping requires more units than are available");
      }
      else
      {
        _CCCL_ASSERT(__prev_nunits >= static_cast<::cuda::std::uint32_t>(__unit_counts_sum),
                     "group_as mapping requires more units than are available");
      }

      if (__prev_unit_rank >= static_cast<::cuda::std::uint32_t>(__unit_counts_sum))
      {
        return _MappingResult::__invalid();
      }
    }

    ::cuda::std::uint32_t __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::uint32_t __i = 0; __i < __curr_ngroups; ++__i)
    {
      const auto __i_count = __unit_counts[__i];
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
};

template <::cuda::std::size_t _StaticGroupCount>
struct __group_as_dynamic_tag;

template <::cuda::std::size_t _StaticGroupCount, bool _IsAlwaysExhaustive>
class group_as<__group_as_dynamic_tag<_StaticGroupCount>, _IsAlwaysExhaustive>
{
  static_assert(_StaticGroupCount != ::cuda::std::dynamic_extent, "group_as requires static number of groups");

  ::cuda::std::uint32_t __unit_counts_[_StaticGroupCount];

public:
  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES(_IsAlwaysExhaustive2)
  _CCCL_DEVICE_API explicit constexpr group_as(
    ::cuda::std::span<const ::cuda::std::uint32_t, _StaticGroupCount> __unit_counts) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _StaticGroupCount; ++__i)
    {
      _CCCL_ASSERT(__unit_counts[__i] > 0, "none of the __unit_counts can be 0");
      __unit_counts_[__i] = __unit_counts[__i];
    }
  }

  _CCCL_TEMPLATE(bool _IsAlwaysExhaustive2 = _IsAlwaysExhaustive)
  _CCCL_REQUIRES((!_IsAlwaysExhaustive2))
  _CCCL_DEVICE_API explicit constexpr group_as(
    ::cuda::std::span<const ::cuda::std::uint32_t, _StaticGroupCount> __unit_counts, const non_exhaustive_t&) noexcept
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::size_t __i = 0; __i < _StaticGroupCount; ++__i)
    {
      _CCCL_ASSERT(__unit_counts[__i] > 0, "none of the __unit_counts can be 0");
      __unit_counts_[__i] = __unit_counts[__i];
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i) noexcept
  {
    if (__i >= _StaticGroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t unit_count(::cuda::std::size_t __i) const noexcept
  {
    if (__i >= _StaticGroupCount)
    {
      _CCCL_THROW(::std::out_of_range, "__i is out of range");
    }
    return __unit_counts_[__i];
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_prev_nunits  = _PrevMappingResult::static_unit_count();
    constexpr auto __static_curr_ngroups = _StaticGroupCount;
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * __static_curr_ngroups)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups,
                       ::cuda::std::dynamic_extent,
                       _PrevMappingResult::is_always_exhaustive() && _IsAlwaysExhaustive,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_nunits      = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();
    constexpr auto __curr_ngroups = static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
    const auto __ngroups          = __prev_mapping_result.group_count() * __curr_ngroups;

    // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
    if constexpr (_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(::cuda::std::accumulate(__unit_counts_, __unit_counts_ + __curr_ngroups, 0u) == __prev_nunits,
                   "group_as mapping _IsAlwaysExhaustive precondition violation");
    }
    else if (__prev_unit_rank >= ::cuda::std::accumulate(__unit_counts_, __unit_counts_ + __curr_ngroups, 0u))
    {
      return _MappingResult::__invalid();
    }

    ::cuda::std::uint32_t __sum = 0;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (::cuda::std::uint32_t __i = 0; __i < __curr_ngroups; ++__i)
    {
      const auto __i_count = __unit_counts_[__i];
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
};

template <::cuda::std::size_t... _StaticUnitCounts>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES
group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _StaticUnitCounts...>&)
  -> group_as<__group_as_static_tag<_StaticUnitCounts...>, true>;

template <::cuda::std::size_t... _StaticUnitCounts>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES
group_as(const ::cuda::std::integer_sequence<::cuda::std::size_t, _StaticUnitCounts...>&, const non_exhaustive_t&)
  -> group_as<__group_as_static_tag<_StaticUnitCounts...>, false>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(
  __is_spannable<_Tp> _CCCL_AND ::cuda::std::
    is_same_v<::cuda::std::uint32_t, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(_Tp& __v)
  -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, true>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(
  __is_spannable<_Tp> _CCCL_AND ::cuda::std::
    is_same_v<::cuda::std::uint32_t, _SpanValueType<decltype(::cuda::std::span(::cuda::std::declval<_Tp&>()))>>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(_Tp& __v, const non_exhaustive_t&)
  -> group_as<__group_as_dynamic_tag<decltype(::cuda::std::span(__v))::extent>, false>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH
