//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_GROUP_VIEW_CUH
#define _CUDA_EXPERIMENTAL___GROUP_GROUP_VIEW_CUH

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
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/__group/queries.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Unit, class _Group>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
__do_group_view_mapping(const _Unit& __unit, const _Group& __group) noexcept
{
  if constexpr (::cuda::std::is_same_v<_Unit, typename _Group::unit_type>)
  {
    return __group.__mapping_result();
  }
  else
  {
    using _GroupMappingResult = typename _Group::__mapping_result_type;
    using _MappingResult =
      __mapping_result<_GroupMappingResult::static_group_count(),
                       ::cuda::experimental::__static_count_query_group<_Unit, _Group>(),
                       _GroupMappingResult::is_always_exhaustive(),
                       _GroupMappingResult::is_always_contiguous()>;

    const auto& __group_mapping_result = __group.__mapping_result();
    return _MappingResult{
      __group_mapping_result.group_count(),
      __group_mapping_result.group_rank(),
      ::cuda::experimental::__count_query_group<::cuda::std::uint32_t, _Unit>(__group),
      ::cuda::experimental::__rank_query_group<::cuda::std::uint32_t, _Unit>(__group),
      __group_mapping_result.lane_mask()};
  }
}

template <class _Unit, class _Group>
using __group_view_mapping_result_t =
  decltype(::cuda::experimental::__do_group_view_mapping(_Unit{}, ::cuda::std::declval<_Group>()));

template <class _Unit, class _Group>
class group_view
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  static_assert(is_group<_Group>);
  static_assert(__unit_same_as_or_below_v<_Unit, typename _Group::unit_type>,
                "unit_type must be same as or below the _Group's unit_type");

  using _MappingResult        = __group_view_mapping_result_t<_Unit, _Group>;
  using _SynchronizerInstance = decltype(::cuda::std::declval<const _Group&>().__synchronizer_instance().view());
  static_assert(__group_mapping_result<_MappingResult>);

  typename _Group::hierarchy_type __hier_;
  _MappingResult __mapping_result_;
  _SynchronizerInstance __synchronizer_instance_;

public:
  using unit_type             = _Unit;
  using level_type            = typename _Group::level_type;
  using hierarchy_type        = typename _Group::hierarchy_type;
  using __mapping_result_type = _MappingResult;

  group_view() = delete;

  _CCCL_HIDE_FROM_ABI group_view(const group_view&) noexcept = default;

  _CCCL_HIDE_FROM_ABI group_view(group_view&&) noexcept = default;

  _CCCL_TEMPLATE(class _Unit2 = _Unit)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_Unit2, typename _Group::unit_type>)
  _CCCL_DEVICE_API group_view(const _Group& __group) noexcept
      : __hier_{__group.hierarchy()}
      , __mapping_result_{::cuda::experimental::__do_group_view_mapping(_Unit{}, __group)}
      , __synchronizer_instance_{__group.__synchronizer_instance().view()}
  {}

  _CCCL_DEVICE_API explicit group_view(const _Unit& __unit, const _Group& __group) noexcept
      : __hier_{__group.hierarchy()}
      , __mapping_result_{::cuda::experimental::__do_group_view_mapping(__unit, __group)}
      , __synchronizer_instance_{__group.__synchronizer_instance().view()}
  {}

  _CCCL_TEMPLATE(class _OtherUnit)
  _CCCL_REQUIRES(__unit_same_as_or_below_v<_Unit, _OtherUnit>)
  _CCCL_DEVICE_API explicit group_view(const _Unit& __unit, const group_view<_OtherUnit, _Group>& __other) noexcept
      : __hier_{__other.hierarchy()}
      , __mapping_result_{::cuda::experimental::__do_group_view_mapping(__unit, __other)}
      , __synchronizer_instance_{__other.__synchronizer_instance()}
  {}

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

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr auto static_count(const _Arg& __arg) noexcept
    -> decltype(_Group::static_count(__arg))
  {
    return _Group::static_count(__arg);
  }

  _CCCL_TEMPLATE(class _Tp, class _Arg)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count_as(const _Arg& __arg) const noexcept
    -> decltype(::cuda::std::declval<const _Group&>().template count_as<_Tp>(__arg))
  {
    return _Group::template __count_as_impl<_Tp>(__mapping_result_, __hier_, __arg);
  }

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _Arg& __arg) const noexcept
    -> decltype(::cuda::std::declval<const _Group&>().count(__arg))
  {
    using _Level = typename _Arg::level_type;
    return _Group::template __count_as_impl<typename _Level::__product_type>(__mapping_result_, __hier_, __arg);
  }

  _CCCL_TEMPLATE(class _Tp, class _Arg)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto rank_as(const _Arg& __arg) const noexcept
    -> decltype(::cuda::std::declval<const _Group&>().template rank_as<_Tp>(__arg))
  {
    return _Group::template __rank_as_impl<_Tp>(__mapping_result_, __hier_, __arg);
  }

  template <class _Arg>
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto rank(const _Arg& __arg) const noexcept
    -> decltype(::cuda::std::declval<const _Group&>().rank(__arg))
  {
    using _Level = typename _Arg::level_type;
    return _Group::template __rank_as_impl<typename _Level::__product_type>(__mapping_result_, __hier_, __arg);
  }
};

_CCCL_TEMPLATE(class _Group)
_CCCL_REQUIRES(is_group<_Group>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Group&) -> group_view<typename _Group::unit_type, _Group>;

template <class _Unit, class _Group>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const group_view<_Unit, _Group>&) -> group_view<_Unit, _Group>;

_CCCL_TEMPLATE(class _Unit, class _Group)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND is_group<_Group>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Unit&, const _Group&) -> group_view<_Unit, _Group>;

_CCCL_TEMPLATE(class _Unit, class _OtherUnit, class _Group)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND __unit_same_as_or_below_v<_Unit, _OtherUnit>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES group_view(const _Unit&, const group_view<_OtherUnit, _Group>&)
  -> group_view<_Unit, _Group>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_GROUP_VIEW_CUH
