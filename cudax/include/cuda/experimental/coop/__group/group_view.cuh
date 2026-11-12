//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_GROUP_VIEW_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_GROUP_VIEW_CUH

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
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/concepts.cuh>
#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/coop/__group/queries.cuh>
#include <cuda/experimental/coop/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Unit, class _Group, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
__do_group_view_mapping(const _Unit&, const _Group& __group, ::cuda::std::index_sequence<_Is...>)
{
  using _GroupMappingResult = typename _Group::__mapping_result_type;
  using _UnitExts =
    decltype(_Unit::extents(typename _Group::unit_type{}, ::cuda::std::declval<typename _Group::hierarchy_type>()));
  constexpr auto __units_per_group_unit = [] {
    ::cuda::std::size_t __count = 1;
    for (::cuda::std::size_t __i = 0; __i < _UnitExts::rank(); ++__i)
    {
      __count *= _UnitExts::static_extent(__i);
    }
    return __count;
  }();
  using _MappingResult =
    __mapping_result<__static_unit_counts<(_GroupMappingResult::static_unit_count(_Is) * __units_per_group_unit)...>,
                     _GroupMappingResult::is_always_exhaustive(),
                     _GroupMappingResult::is_always_contiguous()>;

  const auto& __group_mapping_result = __group.__mapping_result();
  if constexpr (!_GroupMappingResult::is_always_exhaustive())
  {
    if (!__group_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }
  }
  return _MappingResult{__group_mapping_result.group_rank(),
                        ::cuda::experimental::coop::__rank_query_group<::cuda::std::uint32_t, _Unit>(__group),
                        __group_mapping_result.lane_mask()};
}

template <class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto __do_group_view_mapping(const _Unit& __unit, const _Group& __group)
{
  if constexpr (::cuda::std::is_same_v<_Unit, typename _Group::unit_type>)
  {
    return __group.__mapping_result();
  }
  else
  {
    using _GroupMappingResult = typename _Group::__mapping_result_type;
    using _UnitExts =
      decltype(_Unit::extents(typename _Group::unit_type{}, ::cuda::std::declval<typename _Group::hierarchy_type>()));
    if constexpr (!_GroupMappingResult::has_uniform_static_unit_count() && _UnitExts::rank_dynamic() == 0)
    {
      return ::cuda::experimental::coop::__do_group_view_mapping(
        __unit, __group, ::cuda::std::make_index_sequence<_GroupMappingResult::static_group_count()>{});
    }
    else
    {
      using _MappingResult = __mapping_result<
        __uniform_static_unit_count<_GroupMappingResult::static_group_count(),
                                    ::cuda::experimental::coop::__static_count_query_group<_Unit, _Group>()>,
        _GroupMappingResult::is_always_exhaustive(),
        _GroupMappingResult::is_always_contiguous()>;

      const auto& __group_mapping_result = __group.__mapping_result();
      return _MappingResult{
        __group_mapping_result.group_count(),
        __group_mapping_result.group_rank(),
        ::cuda::experimental::coop::__count_query_group<::cuda::std::uint32_t, _Unit>(__group),
        ::cuda::experimental::coop::__rank_query_group<::cuda::std::uint32_t, _Unit>(__group),
        __group_mapping_result.lane_mask()};
    }
  }
}

template <class _Unit, class _Group>
using __group_view_mapping_result_t =
  decltype(::cuda::experimental::coop::__do_group_view_mapping(_Unit{}, ::cuda::std::declval<_Group>()));

template <::cuda::std::size_t _First, ::cuda::std::size_t _Last, class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __with_static_unit_count_range(const _Group& __group, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  using _Unit          = typename _Group::unit_type;

  const auto& __mapping_result = __group.__mapping_result();

  if constexpr (_Last - _First > 1)
  {
    // Rank thresholds keep large collective callbacks in a branch tree instead
    // of encouraging jump-table dispatch from a chain of equality comparisons.
    constexpr auto __mid = _First + (_Last - _First) / 2;
    return (__mapping_result.group_rank() < __mid)
           ? ::cuda::experimental::coop::__with_static_unit_count_range<_First, __mid>(
               __group, ::cuda::std::forward<_Fn>(__fn))
           : ::cuda::experimental::coop::__with_static_unit_count_range<__mid, _Last>(
               __group, ::cuda::std::forward<_Fn>(__fn));
  }
  else
  {
    using _PerGroupMappingResult = ::cuda::experimental::coop::__mapping_result<
      __per_group_count<_MappingResult::static_group_count(), _First, _MappingResult::static_unit_count(_First)>,
      _MappingResult::is_always_exhaustive(),
      _MappingResult::is_always_contiguous()>;

    return ::cuda::std::forward<_Fn>(__fn)(group_view<_Unit, _Group, _PerGroupMappingResult>{
      __group, _PerGroupMappingResult{__mapping_result.unit_rank(), __mapping_result.lane_mask()}});
  }
}

template <class _Group, class _Fn>
_CCCL_DEVICE_API decltype(auto) __with_static_unit_count(const _Group& __group, _Fn&& __fn)
{
  using _MappingResult = typename _Group::__mapping_result_type;
  if constexpr (_MappingResult::has_uniform_static_unit_count())
  {
    return ::cuda::std::forward<_Fn>(__fn)(group_view{__group});
  }
  else
  {
    return ::cuda::experimental::coop::__with_static_unit_count_range<0, _MappingResult::static_group_count()>(
      __group, ::cuda::std::forward<_Fn>(__fn));
  }
}

template <class _Unit, class _Group, class _MappingResult>
class group_view
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  static_assert(group<_Group>);
  static_assert(__unit_same_as_or_below_v<_Unit, typename _Group::unit_type>,
                "unit_type must be same as or below the _Group's unit_type");

  using _SynchronizerInstance _CCCL_NODEBUG = typename _Group::__synchronizer_instance_type;
  static_assert(__group_mapping_result<_MappingResult>);

  typename _Group::hierarchy_type __hier_;
  _MappingResult __mapping_result_;
  _SynchronizerInstance __synchronizer_instance_;

public:
  using unit_type                    = _Unit;
  using level_type                   = typename _Group::level_type;
  using hierarchy_type               = typename _Group::hierarchy_type;
  using __mapping_result_type        = _MappingResult;
  using __synchronizer_instance_type = _SynchronizerInstance;
  using arrival_token                = typename _Group::arrival_token;

  group_view() = delete;

  _CCCL_HIDE_FROM_ABI group_view(const group_view&) = default;

  _CCCL_HIDE_FROM_ABI group_view(group_view&&) = default;

  _CCCL_HIDE_FROM_ABI group_view& operator=(const group_view&) = default;

  _CCCL_HIDE_FROM_ABI group_view& operator=(group_view&&) = default;

  _CCCL_TEMPLATE(class _Unit2 = _Unit)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Unit2, typename _Group::unit_type>)
  _CCCL_DEVICE_API group_view(const _Group& __group)
      : __hier_{__group.hierarchy()}
      , __mapping_result_{::cuda::experimental::coop::__do_group_view_mapping(_Unit{}, __group)}
      , __synchronizer_instance_{__group.__synchronizer_instance()}
  {}

  _CCCL_DEVICE_API explicit group_view(const _Unit& __unit, const _Group& __group)
      : __hier_{__group.hierarchy()}
      , __mapping_result_{::cuda::experimental::coop::__do_group_view_mapping(__unit, __group)}
      , __synchronizer_instance_{__group.__synchronizer_instance()}
  {}

  _CCCL_TEMPLATE(class _OtherUnit, class _OtherMappingResult)
  _CCCL_REQUIRES(__unit_same_as_or_below_v<_Unit, _OtherUnit> _CCCL_AND ::cuda::std::is_same_v<
                 _MappingResult,
                 __group_view_mapping_result_t<_Unit, group_view<_OtherUnit, _Group, _OtherMappingResult>>>)
  _CCCL_DEVICE_API explicit group_view(const _Unit& __unit,
                                       const group_view<_OtherUnit, _Group, _OtherMappingResult>& __other)
      : __hier_{__other.hierarchy()}
      , __mapping_result_{::cuda::experimental::coop::__do_group_view_mapping(__unit, __other)}
      , __synchronizer_instance_{__other.__synchronizer_instance()}
  {}

  // Internal constructor for a view with a specialized mapping result.
  _CCCL_DEVICE_API explicit group_view(const _Group& __group, const _MappingResult& __mapping_result)
      : __hier_{__group.hierarchy()}
      , __mapping_result_{__mapping_result}
      , __synchronizer_instance_{__group.__synchronizer_instance()}
  {}

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

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto static_count(const _Arg& __arg)
    -> decltype(_Group::static_count(__arg))
  {
    return _Group::static_count(__arg);
  }

  template <class _Tp, class _QueryMappingResult, class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  __count_as_impl(const _QueryMappingResult& __mapping_result, const hierarchy_type& __hier, const _Arg& __arg)
    -> decltype(_Group::template __count_as_impl<_Tp>(__mapping_result, __hier, __arg))
  {
    return _Group::template __count_as_impl<_Tp>(__mapping_result, __hier, __arg);
  }

  template <class _Tp, class _QueryMappingResult, class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto
  __rank_as_impl(const _QueryMappingResult& __mapping_result, const hierarchy_type& __hier, const _Arg& __arg)
    -> decltype(_Group::template __rank_as_impl<_Tp>(__mapping_result, __hier, __arg))
  {
    return _Group::template __rank_as_impl<_Tp>(__mapping_result, __hier, __arg);
  }

  _CCCL_TEMPLATE(class _Tp, class _Arg)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count_as(const _Arg& __arg) const
    -> decltype(::cuda::std::declval<const _Group&>().template count_as<_Tp>(__arg))
  {
    return _Group::template __count_as_impl<_Tp>(__mapping_result_, __hier_, __arg);
  }

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _Arg& __arg) const
    -> decltype(::cuda::std::declval<const _Group&>().count(__arg))
  {
    using _Level = typename _Arg::level_type;
    return _Group::template __count_as_impl<typename _Level::__product_type>(__mapping_result_, __hier_, __arg);
  }

  _CCCL_TEMPLATE(class _Tp, class _Arg)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto rank_as(const _Arg& __arg) const
    -> decltype(::cuda::std::declval<const _Group&>().template rank_as<_Tp>(__arg))
  {
    return _Group::template __rank_as_impl<_Tp>(__mapping_result_, __hier_, __arg);
  }

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto rank(const _Arg& __arg) const
    -> decltype(::cuda::std::declval<const _Group&>().rank(__arg))
  {
    using _Level = typename _Arg::level_type;
    return _Group::template __rank_as_impl<typename _Level::__product_type>(__mapping_result_, __hier_, __arg);
  }
};

_CCCL_TEMPLATE(class _Group)
_CCCL_REQUIRES(group<_Group>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Group&)
  -> group_view<typename _Group::unit_type, _Group, __group_view_mapping_result_t<typename _Group::unit_type, _Group>>;

template <class _Unit, class _Group, class _MappingResult>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const group_view<_Unit, _Group, _MappingResult>&)
  -> group_view<_Unit, _Group, _MappingResult>;

_CCCL_TEMPLATE(class _Unit, class _Group)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND group<_Group>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Unit&, const _Group&)
  -> group_view<_Unit, _Group, __group_view_mapping_result_t<_Unit, _Group>>;

_CCCL_TEMPLATE(class _Unit, class _OtherUnit, class _Group, class _MappingResult)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND __unit_same_as_or_below_v<_Unit, _OtherUnit>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Unit&, const group_view<_OtherUnit, _Group, _MappingResult>&)
  -> group_view<_Unit, _Group, __group_view_mapping_result_t<_Unit, group_view<_OtherUnit, _Group, _MappingResult>>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_GROUP_VIEW_CUH
