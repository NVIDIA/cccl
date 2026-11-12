//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_BINARY_PARTITION_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_BINARY_PARTITION_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/lane_mask.h>
#include <cuda/hierarchy>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <class _Callable, class _Unit, class _ParentGroup, class _PrevMappingResult>
[[nodiscard]] _CCCL_DEVICE_API auto __binary_parition_map(
  _Callable& __fn, const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result)
{
  static_assert(::cuda::std::is_same_v<_Unit, thread_level>, "binary_partition can only group threads");
  static_assert(::cuda::std::is_same_v<typename _ParentGroup::level_type, warp_level>,
                "binary_partition can be only used within warp_level");

  constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
  constexpr auto __static_ngroups =
    (__static_prev_ngroups != ::cuda::std::dynamic_extent) ? (__static_prev_ngroups * 2) : ::cuda::std::dynamic_extent;

  using _MappingResult =
    __mapping_result<__uniform_static_unit_count<__static_ngroups, ::cuda::std::dynamic_extent>,
                     _PrevMappingResult::is_always_exhaustive(),
                     false>;

  if constexpr (!_PrevMappingResult::is_always_exhaustive())
  {
    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }
  }

  using _PredicateResult = ::cuda::std::__call_result_t<_Callable&, const _PrevMappingResult&>;
  static_assert(::cuda::std::is_convertible_v<_PredicateResult, bool>,
                "binary_partition requires the callable result to be convertible to bool");

  const auto __pred      = static_cast<bool>(__fn(__prev_mapping_result));
  const auto __prev_mask = __prev_mapping_result.lane_mask().value();

  auto __match_mask = ::__ballot_sync(__prev_mask, __pred);
  if (!__pred)
  {
    __match_mask = (~__match_mask) & __prev_mask;
  }
  return _MappingResult{
    __prev_mapping_result.group_count() * 2,
    __prev_mapping_result.group_rank() + ((__pred) ? __prev_mapping_result.group_count() : 0u),
    static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__match_mask)),
    static_cast<::cuda::std::uint32_t>(::cuda::std::popcount(__match_mask & ::cuda::ptx::get_sreg_lanemask_lt())),
    ::cuda::device::lane_mask{__match_mask}};
}

template <class _Fn>
class binary_partition
{
  static_assert(::cuda::std::is_move_constructible_v<_Fn>, "_Fn must be move constructible");

  _Fn __fn_;

public:
  _CCCL_DEVICE_API explicit binary_partition(_Fn __fn)
      : __fn_(::cuda::std::move(__fn))
  {}

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<_Fn&, const _PrevMappingResult&>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__binary_parition_map(__fn_, __unit, __parent, __prev_mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<const _Fn&, const _PrevMappingResult&>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit& __unit, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__binary_parition_map(__fn_, __unit, __parent, __prev_mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<_Fn&, const _PrevMappingResult&>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup& __parent,
      const _PrevMappingResult& __prev_mapping_result)
  {
    return ::cuda::experimental::coop::__binary_parition_map(__fn_, __unit, __parent, __prev_mapping_result);
  }

  _CCCL_TEMPLATE(class _Unit, class _ParentGroup, class _PrevMappingResult)
  _CCCL_REQUIRES(::cuda::std::__is_callable_v<const _Fn&, const _PrevMappingResult&>)
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const non_exhaustive_t&,
      const _Unit& __unit,
      const _ParentGroup& __parent,
      const _PrevMappingResult& __prev_mapping_result) const
  {
    return ::cuda::experimental::coop::__binary_parition_map(__fn_, __unit, __parent, __prev_mapping_result);
  }
};

template <class _PredFn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES binary_partition(_PredFn) -> binary_partition<_PredFn>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_BINARY_PARTITION_CUH
