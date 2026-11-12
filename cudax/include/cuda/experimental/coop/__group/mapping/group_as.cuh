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

#include <cuda/__numeric/narrow.h>
#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__numeric/accumulate.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/all.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/constant_wrapper.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/common.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <bool _Exhaustive, class _Unit, class _PrevMappingResult, class _Range>
[[nodiscard]] _CCCL_DEVICE_API auto
__group_as_map(const _Unit&, const _PrevMappingResult& __prev_mapping_result, _Range& __unit_counts)
{
  using ::cuda::std::dynamic_extent;

  constexpr auto __prev_static_group_count = _PrevMappingResult::static_group_count();
  constexpr auto __curr_static_group_count = ::cuda::std::ranges::__static_size_v<_Range>;
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

  const auto __prev_nunits    = __prev_mapping_result.unit_count();
  const auto __prev_unit_rank = __prev_mapping_result.unit_rank();
  const auto __curr_ngroups   = static_cast<::cuda::std::uint32_t>(::cuda::std::ranges::size(__unit_counts));
  const auto __ngroups        = __prev_mapping_result.group_count() * __curr_ngroups;

  // If the mapping is exhaustive, check the preconditions, otherwise remove the last partial group.
  if constexpr (_Exhaustive)
  {
    _CCCL_ASSERT(
      ::cuda::std::accumulate(::cuda::std::ranges::begin(__unit_counts), ::cuda::std::ranges::end(__unit_counts), 0u)
        == __prev_nunits,
      "group_as mapping would produce units without a group during exhaustive mapping");
  }
  else
  {
    _CCCL_ASSERT(
      ::cuda::std::accumulate(::cuda::std::ranges::begin(__unit_counts), ::cuda::std::ranges::end(__unit_counts), 0u)
        <= __prev_nunits,
      "group_as mapping requires more units than are available");

    if (__prev_unit_rank >= ::cuda::std::accumulate(
          ::cuda::std::ranges::begin(__unit_counts), ::cuda::std::ranges::end(__unit_counts), 0u))
    {
      return _MappingResult::__invalid();
    }
  }

  ::cuda::std::uint32_t __sum = 0;
  auto __it                   = ::cuda::std::ranges::begin(__unit_counts);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (::cuda::std::uint32_t __i = 0; __i < __curr_ngroups; ++__i, ++__it)
  {
    const auto __i_count = static_cast<::cuda::std::uint32_t>(*__it);
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
  _CCCL_DEVICE_API explicit group_as(_Range&& __unit_counts)
      : __unit_counts_{::cuda::std::ranges::views::all(::cuda::std::forward<_Range>(__unit_counts))}
  {}

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__group_as_map<true>(__unit, __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES((!::cuda::std::ranges::range<const _RangeView>) )
  [[nodiscard]] _CCCL_DEVICE_API auto map(
    const non_exhaustive_t&, const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__group_as_map<false>(__unit, __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__group_as_map<true>(__unit, __prev_mapping_result, __unit_counts_);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::ranges::range<const _RangeView>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup&,
      const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__group_as_map<false>(__unit, __prev_mapping_result, __unit_counts_);
  }
};

// Materialize a compile-time range into a sequence without requiring random-access iterators.
template <class _Wrapper, ::cuda::std::size_t... _Is>
[[nodiscard]]
_CCCL_DEVICE_API constexpr auto __group_as_sequence(::cuda::std::index_sequence<_Is...>) -> ::cuda::std::index_sequence<
  ::cuda::narrow<::cuda::std::size_t>(*::cuda::std::next(::cuda::std::ranges::begin(_Wrapper::value), _Is))...>;

template <class _Wrapper>
using __group_as_sequence_t = decltype(::cuda::experimental::coop::__group_as_sequence<_Wrapper>(
  ::cuda::std::make_index_sequence<::cuda::std::ranges::size(_Wrapper::value)>{}));

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
    using _Result =
      __mapping_result<__static_unit_counts<::cuda::std::array<::cuda::std::size_t, sizeof...(_Counts)>{
                         _Counts...}[_Is % sizeof...(_Counts)]...>,
                       _Exhaustive,
                       _PrevMappingResult::is_always_contiguous()>;
    if (!__result.is_valid())
    {
      return _Result::__invalid();
    }
    return _Result{__result.group_rank(), __result.unit_rank(), __result.lane_mask()};
  }

  template <bool _Exhaustive, class _Unit, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto __do_map(const _Unit& __unit, const _PrevMappingResult& __prev) const
  {
    // Check every statically known input size, including non-uniform previous groups.
    constexpr bool __compatible = [] {
      if constexpr (_PrevMappingResult::has_uniform_static_unit_count())
      {
        constexpr auto __n = _PrevMappingResult::static_unit_count();
        return __n == ::cuda::std::dynamic_extent || (_Exhaustive ? __n == __sum : __n >= __sum);
      }
      else if constexpr (_PrevMappingResult::static_group_count() != ::cuda::std::dynamic_extent)
      {
        for (::cuda::std::size_t __i = 0; __i < _PrevMappingResult::static_group_count(); ++__i)
        {
          const auto __n = _PrevMappingResult::static_unit_count(__i);
          if (_Exhaustive ? __n != __sum : __n < __sum)
          {
            return false;
          }
        }
      }
      return true;
    }();
    static_assert(__compatible, "group_as counts are incompatible with the previous static unit counts");

    constexpr ::cuda::std::uint32_t __counts[]{static_cast<::cuda::std::uint32_t>(_Counts)...};
    const auto __result = ::cuda::experimental::coop::__group_as_map<_Exhaustive>(__unit, __prev, __counts);

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
  _CCCL_TEMPLATE(class _Int, _Int... _Values)
  _CCCL_REQUIRES(::cuda::std::is_same_v<::cuda::std::index_sequence<::cuda::narrow<::cuda::std::size_t>(_Values)...>,
                                        ::cuda::std::index_sequence<_Counts...>>)
  _CCCL_DEVICE_API explicit constexpr group_as(const ::cuda::std::integer_sequence<_Int, _Values...>&) {}

  _CCCL_TEMPLATE(auto _Range, class _Tp)
  _CCCL_REQUIRES(::cuda::std::is_same_v<__group_as_sequence_t<::cuda::std::constant_wrapper<_Range, _Tp>>,
                                        ::cuda::std::index_sequence<_Counts...>>)
  _CCCL_DEVICE_API explicit constexpr group_as(const ::cuda::std::constant_wrapper<_Range, _Tp>&) {}

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev) const
  {
    return __do_map<true>(__unit, __prev);
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&, const _Unit& __unit, const _ParentGroup&, const _PrevMappingResult& __prev) const
  {
    return __do_map<false>(__unit, __prev);
  }
};

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<_Range> _CCCL_AND ::cuda::std::ranges::viewable_range<_Range>
    _CCCL_AND ::cuda::std::is_convertible_v<::cuda::std::ranges::range_value_t<_Range>, ::cuda::std::uint32_t>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(_Range&&) -> group_as<::cuda::std::ranges::views::all_t<_Range>>;

template <class _Int, _Int... _Counts>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(const ::cuda::std::integer_sequence<_Int, _Counts...>&)
  -> group_as<::cuda::std::index_sequence<::cuda::narrow<::cuda::std::size_t>(_Counts)...>>;

_CCCL_TEMPLATE(auto _Range, class _Tp)
_CCCL_REQUIRES(
  ::cuda::std::ranges::sized_range<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>
    _CCCL_AND ::cuda::std::ranges::forward_range<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>
      _CCCL_AND ::cuda::std::is_convertible_v<
        ::cuda::std::ranges::range_value_t<decltype((::cuda::std::constant_wrapper<_Range, _Tp>::value))>,
        ::cuda::std::size_t>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_as(const ::cuda::std::constant_wrapper<_Range, _Tp>&)
  -> group_as<__group_as_sequence_t<::cuda::std::constant_wrapper<_Range, _Tp>>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_GROUP_AS_CUH
