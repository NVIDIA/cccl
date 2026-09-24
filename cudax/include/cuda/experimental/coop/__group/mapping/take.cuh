//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_TAKE_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_TAKE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/integral_constant_like.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/coop/__group/fwd.cuh>
#include <cuda/experimental/coop/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental::coop
{
template <::cuda::std::size_t _StaticUnitCount>
class take
{
  static_assert(::cuda::std::in_range<::cuda::std::uint32_t>(_StaticUnitCount),
                "_StaticUnitCount must be within uint32_t range");

public:
  _CCCL_DEVICE_API constexpr explicit take(::cuda::std::uint32_t __unit_count) noexcept
  {
    _CCCL_ASSERT(__unit_count == _StaticUnitCount, "__unit_count must be same as _StaticUnitCount");
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_nunits = _PrevMappingResult::static_unit_count();

    using _MappingResult =
      __mapping_result<_PrevMappingResult::static_group_count(),
                       _StaticUnitCount,
                       _PrevMappingResult::is_always_exhaustive() && (__static_prev_nunits == _StaticUnitCount),
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_units_count = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();

    if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
    {
      static_assert(__static_prev_nunits >= _StaticUnitCount,
                    "take mapping requires the previous mapping result to have at least _PrevMappingResult units");
    }
    else
    {
      _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__prev_units_count, _StaticUnitCount),
                   "take mapping requires the previous mapping result to have at least _PrevMappingResult units");
    }

    if (::cuda::std::cmp_greater_equal(__prev_unit_rank, static_cast<::cuda::std::uint32_t>(_StaticUnitCount)))
    {
      return _MappingResult::__invalid();
    }

    const auto __group_count = __prev_mapping_result.group_count();
    const auto __group_rank  = __prev_mapping_result.group_rank();
    const auto __unit_count  = static_cast<::cuda::std::uint32_t>(_StaticUnitCount);
    const auto __unit_rank   = __prev_unit_rank;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __unit_count, __unit_rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__group_count, __group_rank, __unit_count, __unit_rank, __lane_mask};
  }
};

template <>
class take<::cuda::std::dynamic_extent>
{
  ::cuda::std::uint32_t __unit_count_;

public:
  _CCCL_DEVICE_API constexpr explicit take(::cuda::std::uint32_t __unit_count) noexcept
      : __unit_count_{__unit_count}
  {}

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    using _MappingResult =
      __mapping_result<_PrevMappingResult::static_group_count(),
                       ::cuda::std::dynamic_extent,
                       false,
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::__invalid();
    }

    const auto __prev_units_count = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();

    _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__prev_units_count, __unit_count_),
                 "take mapping requires the previous mapping result to have at least _PrevMappingResult units");

    if (::cuda::std::cmp_greater_equal(__prev_unit_rank, __unit_count_))
    {
      return _MappingResult::__invalid();
    }

    const auto __group_count = __prev_mapping_result.group_count();
    const auto __group_rank  = __prev_mapping_result.group_rank();
    const auto __unit_count  = __unit_count_;
    const auto __unit_rank   = __prev_unit_rank;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::coop::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __unit_count, __unit_rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__group_count, __group_rank, __unit_count, __unit_rank, __lane_mask};
  }
};

template <class _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES take(_Tp) -> take<::cuda::std::__maybe_static_ext<_Tp>>;
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_TAKE_CUH
