//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_BINARY_PARTITION_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_BINARY_PARTITION_CUH

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
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/is_move_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Fn>
class binary_partition
{
  static_assert(::cuda::std::is_move_constructible_v<_Fn>, "_Fn must be move constructible");

  _Fn __fn_;

public:
  _CCCL_DEVICE_API explicit binary_partition(_Fn __fn) noexcept(::cuda::std::is_nothrow_move_constructible_v<_Fn>)
      : __fn_(::cuda::std::move(__fn))
  {}

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup& __parent, const _PrevMappingResult& __prev_mapping_result) noexcept(
    ::cuda::std::is_nothrow_invocable_v<_Fn, const _PrevMappingResult&>)
  {
    static_assert(::cuda::std::is_same_v<_Unit, thread_level>, "binary_partition can only group threads");
    static_assert(::cuda::std::is_same_v<typename _ParentGroup::level_type, warp_level>,
                  "binary_partition can be only used within warp_level");

    constexpr auto __static_prev_ngroups = _PrevMappingResult::static_group_count();
    constexpr auto __static_ngroups =
      (__static_prev_ngroups != ::cuda::std::dynamic_extent)
        ? (__static_prev_ngroups * 2)
        : ::cuda::std::dynamic_extent;

    using _MappingResult =
      __mapping_result<__static_ngroups, ::cuda::std::dynamic_extent, _PrevMappingResult::is_always_exhaustive(), false>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __pred      = static_cast<bool>(__fn_(__prev_mapping_result));
    const auto __prev_mask = __prev_mapping_result.lane_mask().value();

    auto __match_mask = ::__ballot_sync(__prev_mask, __pred);
    if (!__pred)
    {
      __match_mask = (~__match_mask) & __prev_mask;
    }
    return _MappingResult{
      __prev_mapping_result.group_count() * 2,
      __prev_mapping_result.group_rank() + ((__pred) ? __prev_mapping_result.group_count() : 0u),
      static_cast<unsigned>(::cuda::std::popcount(__match_mask)),
      static_cast<unsigned>(::cuda::std::popcount(__match_mask & ::cuda::ptx::get_sreg_lanemask_lt())),
      ::cuda::device::lane_mask{__match_mask}};
  }
};

template <class _PredFn>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES binary_partition(_PredFn) -> binary_partition<_PredFn>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_BINARY_PARTITION_CUH
