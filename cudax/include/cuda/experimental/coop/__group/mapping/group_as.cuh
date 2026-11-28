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

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/range_to_sequence.cuh>

#include <cuda/std/__cccl/prologue.h>

#ifndef _CCCL_DOXYGEN_INVOKED

namespace cuda::experimental::coop
{
// The group rank and first unit are relative to the range searched. A zero count denotes a unit outside the partition.
struct __group_as_location
{
  ::cuda::std::uint32_t __group_rank_;
  ::cuda::std::uint32_t __first_unit_;
  ::cuda::std::uint32_t __unit_count_;
};

template <::cuda::std::size_t... _Counts>
struct __group_as_search
{
  // Each search node uses two of these compile-time offsets to size its left subtree.
  static constexpr auto __offsets = [] {
    ::cuda::std::array<::cuda::std::uint32_t, sizeof...(_Counts) + 1> __result{};
    constexpr ::cuda::std::uint32_t __sizes[]{static_cast<::cuda::std::uint32_t>(_Counts)...};
    for (::cuda::std::size_t __i = 0; __i < sizeof...(_Counts); ++__i)
    {
      __result[__i + 1] = __result[__i] + __sizes[__i];
    }
    return __result;
  }();

  template <::cuda::std::size_t _First, ::cuda::std::size_t _Last>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr __group_as_location __find(::cuda::std::uint32_t __unit) noexcept
  {
    if constexpr (_Last - _First == 1)
    {
      constexpr auto __size = __offsets[_Last] - __offsets[_First];
      return {0, 0, __size};
    }
    else
    {
      constexpr auto __mid        = _First + ((_Last - _First) / 2);
      constexpr auto __left_units = __offsets[__mid] - __offsets[_First];
      if constexpr (__left_units != 0)
      {
        if (__unit < __left_units)
        {
          return __find<_First, __mid>(__unit);
        }
      }
      auto __right = __find<__mid, _Last>(__unit - __left_units);
      __right.__group_rank_ += static_cast<::cuda::std::uint32_t>(__mid - _First);
      __right.__first_unit_ += __left_units;
      return __right;
    }
  }
};

template <bool _Exhaustive>
[[nodiscard]] _CCCL_API constexpr bool
__group_as_fits(::cuda::std::size_t __total, ::cuda::std::size_t __available) noexcept
{
  return _Exhaustive ? __total == __available : __total <= __available;
}

template <bool _Exhaustive, class _Range, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API __group_as_location
__group_as_find(_Range& __counts, ::cuda::std::uint32_t __ngroups, const _PrevMappingResult& __prev)
{
  if constexpr (::cuda::std::ranges::common_range<_Range>)
  {
    _CCCL_ASSERT(
      ::cuda::experimental::coop::__group_as_fits<_Exhaustive>(
        ::cuda::std::accumulate(::cuda::std::ranges::begin(__counts), ::cuda::std::ranges::end(__counts), 0u),
        __prev.unit_count()),
      "group_as counts are incompatible with the previous unit count");
  }

  const auto __unit                  = __prev.unit_rank();
  ::cuda::std::uint32_t __first_unit = 0;
  auto __it                          = ::cuda::std::ranges::begin(__counts);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::uint32_t __i = 0; __i < __ngroups; ++__i, ++__it)
  {
    const auto __count = static_cast<::cuda::std::uint32_t>(*__it);
    if (__unit < __first_unit + __count)
    {
      return {__i, __first_unit, __count};
    }
    __first_unit += __count;
  }
  if constexpr (!_Exhaustive)
  {
    return {};
  }
  _CCCL_UNREACHABLE();
}

template <bool _Exhaustive, ::cuda::std::size_t... _Counts, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API __group_as_location
__group_as_find(::cuda::std::index_sequence<_Counts...>, ::cuda::std::uint32_t, const _PrevMappingResult& __prev)
{
  constexpr auto __total = (_Counts + ...);
  _CCCL_ASSERT(::cuda::experimental::coop::__group_as_fits<_Exhaustive>(__total, __prev.unit_count()),
               "group_as counts are incompatible with the previous unit count");
  if constexpr (__total != 0)
  {
    const auto __unit = __prev.unit_rank();
    if (_Exhaustive || __unit < __total)
    {
      return __group_as_search<_Counts...>::template __find<0, sizeof...(_Counts)>(__unit);
    }
  }
  return {};
}

template <bool _Exhaustive, class _Unit, class _Bound, class _PrevMappingResult, class _Partition>
[[nodiscard]] _CCCL_DEVICE_API auto
__group_as_map(const _PrevMappingResult& __prev_mapping_result, _Partition& __partition)
{
  using ::cuda::std::dynamic_extent;

  constexpr auto __prev_static_group_count = _PrevMappingResult::static_group_count();
  constexpr auto __curr_static_group_count = ::cuda::std::ranges::__static_size_v<_Partition>;
  constexpr auto __static_group_count =
    (__prev_static_group_count != dynamic_extent && __curr_static_group_count != dynamic_extent)
      ? __prev_static_group_count * __curr_static_group_count
      : dynamic_extent;

  using _MappingResult =
    __mapping_result<__uniform_static_unit_count<__static_group_count, dynamic_extent>,
                     _Exhaustive,
                     _PrevMappingResult::is_always_contiguous()>;

  if constexpr (!_PrevMappingResult::is_always_exhaustive())
  {
    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }
  }

  const auto __curr_ngroups = static_cast<::cuda::std::uint32_t>(::cuda::std::ranges::size(__partition));
  const auto __location =
    ::cuda::experimental::coop::__group_as_find<_Exhaustive>(__partition, __curr_ngroups, __prev_mapping_result);
  if (!_Exhaustive && __location.__unit_count_ == 0)
  {
    return _MappingResult::__invalid();
  }

  const auto __ngroups    = __prev_mapping_result.group_count() * __curr_ngroups;
  const auto __group_rank = (__prev_mapping_result.group_rank() * __curr_ngroups) + __location.__group_rank_;
  const auto __rank       = __prev_mapping_result.unit_rank() - __location.__first_unit_;
  const auto __lane_mask =
    ::cuda::experimental::coop::__make_lane_mask_for_n<_Unit, _Bound, _PrevMappingResult::is_always_contiguous()>(
      __prev_mapping_result.lane_mask(), __location.__unit_count_, __rank);
  return _MappingResult{__ngroups, __group_rank, __location.__unit_count_, __rank, __lane_mask};
}

template <class _RangeView>
class group_as
{
  static_assert(::cuda::std::ranges::sized_range<_RangeView>, "_RangeView must be a sized range");
  static_assert(::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_RangeView>, ::cuda::std::uint32_t>,
                "_RangeView's value_type must be convertible to uint32_t");

  _RangeView __unit_counts_;

public:
  // The viewable-range constraint excludes group_as itself.
  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_RangeView, ::cuda::std::ranges::views::all_t<_Range>>)
  _CCCL_DEVICE_API explicit group_as(_Range&& __unit_counts) // NOLINT(bugprone-forwarding-reference-overload)
      : __unit_counts_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__unit_counts))}
  {}

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__group_as_map<true, _Unit, typename _ParentGroup::boundary_level>(
      __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&, const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__group_as_map<false, _Unit, typename _ParentGroup::boundary_level>(
      __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__group_as_map<true, _Unit, typename _ParentGroup::boundary_level>(
      __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&, const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__group_as_map<false, _Unit, typename _ParentGroup::boundary_level>(
      __prev_mapping_result, __unit_counts_);
  }
};

template <::cuda::std::size_t... _Counts>
class group_as<::cuda::std::index_sequence<_Counts...>>
{
  static_assert(sizeof...(_Counts) > 0, "group_as requires at least one group");
  static_assert((::cuda::std::in_range<::cuda::std::uint32_t>(_Counts) && ...),
                "group_as counts must be representable as uint32_t");
  static constexpr ::cuda::std::size_t __sum = (::cuda::std::size_t{0} + ... + _Counts);
  static_assert(::cuda::std::in_range<::cuda::std::uint32_t>(__sum), "group_as count sum must fit in uint32_t");

  template <bool _Exhaustive, class _PrevMappingResult, class _DynamicResult, ::cuda::std::size_t... _Is>
  [[nodiscard]] _CCCL_DEVICE_API static auto
  __make_non_uniform_mapping_result(const _DynamicResult& __result, ::cuda::std::index_sequence<_Is...>)
  {
    constexpr ::cuda::std::size_t __counts[]{_Counts...};
    using _Result = __mapping_result<__static_unit_counts<__counts[_Is % sizeof...(_Counts)]...>,
                                     _Exhaustive,
                                     _PrevMappingResult::is_always_contiguous()>;
    if (!__result.is_valid())
    {
      return _Result::__invalid();
    }
    return _Result{__result.group_rank(), __result.unit_rank(), __result.lane_mask()};
  }

  template <bool _Exhaustive, class _Unit, class _Bound, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto __do_map(const _PrevMappingResult& __prev) const
  {
    // Check every statically known input size, including non-uniform previous groups.
    constexpr bool __compatible = [] {
      if constexpr (_PrevMappingResult::has_uniform_static_unit_count())
      {
        constexpr auto __n = _PrevMappingResult::static_unit_count();
        return __n == ::cuda::std::dynamic_extent
            || ::cuda::experimental::coop::__group_as_fits<_Exhaustive>(__sum, __n);
      }
      else if constexpr (_PrevMappingResult::static_group_count() != ::cuda::std::dynamic_extent)
      {
        for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
        {
          const auto __n = _PrevMappingResult::static_unit_count(__i);
          if (!::cuda::experimental::coop::__group_as_fits<_Exhaustive>(__sum, __n))
          {
            return false;
          }
        }
      }
      return true;
    }();
    static_assert(__compatible, "group_as counts are incompatible with the previous static unit counts");

    constexpr ::cuda::std::index_sequence<_Counts...> __partition{};
    const auto __result = ::cuda::experimental::coop::__group_as_map<_Exhaustive, _Unit, _Bound>(__prev, __partition);

    if constexpr (_PrevMappingResult::static_group_count() != ::cuda::std::dynamic_extent)
    {
      return __make_non_uniform_mapping_result<_Exhaustive, _PrevMappingResult>(
        __result, ::cuda::std::make_index_sequence<_PrevMappingResult::static_group_count() * sizeof...(_Counts)>{});
    }
    else
    {
      return __result;
    }
  }

public:
  _CCCL_TEMPLATE(class _StaticRange)
  _CCCL_REQUIRES(::cuda::std::is_same_v<__to_index_sequence_t<_StaticRange>, ::cuda::std::index_sequence<_Counts...>>)
  _CCCL_DEVICE_API explicit constexpr group_as(const _StaticRange&) noexcept {}

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev) const
  {
    return __do_map<true, _Unit, typename _ParentGroup::boundary_level>(__prev);
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&, const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev) const
  {
    return __do_map<false, _Unit, typename _ParentGroup::boundary_level>(__prev);
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<_Range> _CCCL_AND ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, ::cuda::std::uint32_t>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(_Range&&) -> group_as<::cuda::std::ranges::views::all_t<_Range>>;

template <class _StaticRange>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(const _StaticRange&) -> group_as<__to_index_sequence_t<_StaticRange>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH
