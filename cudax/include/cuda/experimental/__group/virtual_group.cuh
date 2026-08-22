//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_VIRTUAL_GROUP_CUH
#define _CUDA_EXPERIMENTAL___GROUP_VIRTUAL_GROUP_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bitmask.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__hierarchy/queries/count.h>
#include <cuda/__hierarchy/queries/rank.h>
#include <cuda/barrier>
#include <cuda/hierarchy>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>
#include <cuda/experimental/__group/this_group.cuh>
#include <cuda/experimental/__group/traits.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Unit, class _ParentGroup, class _Mapping>
[[nodiscard]] _CCCL_DEVICE_API constexpr auto
__do_group_mapping(const _Unit& __unit, const _ParentGroup& __parent, const _Mapping& __mapping) noexcept
{
  using _ParentMappingResult = typename _ParentGroup::__mapping_result_type;
  using _InitMappingResult =
    __mapping_result</*initial static group count*/ 1,
                     ::cuda::experimental::__static_count_query_group<_Unit, _ParentGroup>(),
                     _ParentMappingResult::is_always_exhaustive(),
                     _ParentMappingResult::is_always_contiguous()>;
  const _InitMappingResult __init_mapping_result{
    /*initial group count*/ 1,
    /*initial group rank*/ 0,
    ::cuda::experimental::__count_query_group<::cuda::std::uint32_t, _Unit>(__parent),
    ::cuda::experimental::__rank_query_group<::cuda::std::uint32_t, _Unit>(__parent),
    __parent.__mapping_result().lane_mask()};

  const auto __mapping_result = __mapping.map(__unit, __parent, __init_mapping_result);
  if (__mapping_result.is_valid())
  {
    _CCCL_ASSERT(__mapping_result.group_rank() < __mapping_result.group_count(), "invalid group rank");
    _CCCL_ASSERT(__mapping_result.unit_rank() < __mapping_result.unit_count(), "invalid unit rank");

    if constexpr (::cuda::std::is_same_v<_Unit, thread_level>)
    {
      _CCCL_ASSERT(
        (__mapping_result.lane_mask() & ::cuda::device::lane_mask::this_lane()) != ::cuda::device::lane_mask::none(),
        "invalid lane mask - this lane must be contained in the lane mask");
      _CCCL_ASSERT(::cuda::std::popcount(__mapping_result.lane_mask().value()) <= __mapping_result.unit_count(),
                   "invalid lane mask - too many lanes are set in the lane mask");
    }
    else
    {
      _CCCL_ASSERT(__mapping_result.lane_mask() == ::cuda::device::lane_mask::all(),
                   "invalid lane mask - must be equal to cuda::device::lane_mask::all() when _Unit is not "
                   "cuda::thread_level");
    }
  }
  return __mapping_result;
}

template <class _Unit, class _ParentGroup, class _Mapping>
using __group_mapping_result_t = decltype(::cuda::experimental::__do_group_mapping(
  ::cuda::std::declval<const _Unit&>(),
  ::cuda::std::declval<const _ParentGroup&>(),
  ::cuda::std::declval<const _Mapping&>()));

template <class _Unit, class _ParentGroup, class _MappingResult>
class virtual_group
{
  static_assert(__is_hierarchy_level_v<_Unit>);
  static_assert(is_group<_ParentGroup>);
  static_assert(__unit_same_as_or_below_v<_Unit, typename _ParentGroup::unit_type>,
                "unit_type must be same as or below _ParentGroup's unit_type");

  using _Hierarchy            = typename _ParentGroup::hierarchy_type;
  using _ParentMappingResult  = typename _ParentGroup::__mapping_result_type;
  using _SynchronizerInstance = decltype(::cuda::std::declval<const _ParentGroup&>().__synchronizer_instance().view());
  static_assert(__group_mapping_result<_MappingResult>);

  _Hierarchy __hier_;
  _MappingResult __mapping_result_;
  _SynchronizerInstance __synchronizer_instance_;

public:
  using unit_type             = _Unit;
  using level_type            = typename _ParentGroup::level_type;
  using hierarchy_type        = _Hierarchy;
  using __mapping_result_type = _MappingResult;

  _CCCL_TEMPLATE(class _Mapping)
  _CCCL_REQUIRES(::cuda::std::is_same_v<_MappingResult, __group_mapping_result_t<_Unit, _ParentGroup, _Mapping>>)
  _CCCL_DEVICE_API explicit virtual_group(
    const _Unit& __unit, const _ParentGroup& __parent, const _Mapping& __mapping) noexcept
      : __hier_{__parent.hierarchy()}
      , __mapping_result_{::cuda::experimental::__do_group_mapping(__unit, __parent, __mapping)}
      , __synchronizer_instance_{__parent.__synchronizer_instance().view()}
  {
    // todo(dabayer): Remove this if we allow non-exhaustive virtual groups.
    _CCCL_ASSERT(__mapping_result_.is_valid(), "virtual_group requires all units to be part of the group");
  }

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

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count(const _ParentGroup&) noexcept
  {
    return _MappingResult::static_group_count();
  }

  template <class _Tp, class _QueryMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __count_as_impl(const _QueryMappingResult& __mapping_result, const _Hierarchy&, const _ParentGroup&) noexcept
  {
    return static_cast<_Tp>(__mapping_result.group_count());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp count_as(const _ParentGroup& __parent) const noexcept
  {
    return __count_as_impl<_Tp>(__mapping_result_, __hier_, __parent);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr auto count(const _ParentGroup& __parent) const noexcept
  {
    return __count_as_impl<typename level_type::__product_type>(__mapping_result_, __hier_, __parent);
  }

  template <class _Tp, class _QueryMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API static constexpr _Tp
  __rank_as_impl(const _QueryMappingResult& __mapping_result, const _Hierarchy&, const _ParentGroup&) noexcept
  {
    return static_cast<_Tp>(__mapping_result.group_rank());
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(::cuda::std::__cccl_is_integer_v<_Tp>)
  [[nodiscard]] _CCCL_DEVICE_API _Tp rank_as(const _ParentGroup& __parent) const noexcept
  {
    return __rank_as_impl<_Tp>(__mapping_result_, __hier_, __parent);
  }

  [[nodiscard]] _CCCL_DEVICE_API auto rank(const _ParentGroup& __parent) const noexcept
  {
    return __rank_as_impl<typename level_type::__product_type>(__mapping_result_, __hier_, __parent);
  }
};

_CCCL_TEMPLATE(class _Unit,
               class _ParentGroup,
               class _Mapping,
               class _MappingResult = __group_mapping_result_t<_Unit, _ParentGroup, _Mapping>)
_CCCL_REQUIRES(__is_hierarchy_level_v<_Unit> _CCCL_AND is_group<_ParentGroup> _CCCL_AND
                 __unit_same_as_or_below_v<_Unit, typename _ParentGroup::unit_type>)
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES virtual_group(const _Unit&, const _ParentGroup&, const _Mapping&)
  -> virtual_group<_Unit, _ParentGroup, _MappingResult>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_VIRTUAL_GROUP_CUH
