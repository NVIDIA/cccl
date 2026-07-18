//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_TAKE_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_TAKE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/span>

#include <cuda/experimental/__group/fwd.cuh>
#include <cuda/experimental/__group/mapping/mapping_result.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <::cuda::std::size_t _UnitCount>
class take
{
  static_assert(::cuda::std::in_range<unsigned>(_UnitCount), "_UnitCount must be within uint32_t range");

public:
  _CCCL_HIDE_FROM_ABI explicit take() = default;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count() noexcept
  {
    return _UnitCount;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr unsigned unit_count() const noexcept
  {
    return unsigned{_UnitCount};
  }

  template <class _Unit, class _ParentGroup, class _PrevMappingResult>
  [[nodiscard]] _CCCL_DEVICE_API auto
  map(const _Unit&, const _ParentGroup&, const _PrevMappingResult& __prev_mapping_result) const noexcept
  {
    constexpr auto __static_prev_nunits = _PrevMappingResult::static_unit_count();

    using _MappingResult =
      __mapping_result<_PrevMappingResult::static_group_count(),
                       _UnitCount,
                       _PrevMappingResult::is_always_exhaustive() && (__static_prev_nunits == _UnitCount),
                       _PrevMappingResult::is_always_contiguous()>;

    if (!__prev_mapping_result.is_valid())
    {
      return _MappingResult::invalid();
    }

    const auto __prev_units_count = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();

    if constexpr (__static_prev_nunits != ::cuda::std::dynamic_extent)
    {
      static_assert(__static_prev_nunits >= _UnitCount,
                    "take mapping requires the previous mapping result to have at least _PrevMappingResult units");
    }
    else
    {
      _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__prev_units_count, _UnitCount),
                   "take mapping requires the previous mapping result to have at least _PrevMappingResult units");
    }

    if (::cuda::std::cmp_greater_equal(__prev_unit_rank, static_cast<unsigned>(_UnitCount)))
    {
      return _MappingResult::invalid_with_group_count(__prev_mapping_result.group_count());
    }

    const auto __group_count = __prev_mapping_result.group_count();
    const auto __group_rank  = __prev_mapping_result.group_rank();
    const auto __unit_count  = static_cast<unsigned>(_UnitCount);
    const auto __unit_rank   = __prev_unit_rank;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __unit_count, __unit_rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__group_count, __group_rank, __unit_count, __unit_rank, __lane_mask};
  }
};

template <>
class take<::cuda::std::dynamic_extent>
{
  unsigned __unit_count_{0};

public:
  _CCCL_HIDE_FROM_ABI explicit take() = default;

  _CCCL_DEVICE_API constexpr explicit take(unsigned __unit_count) noexcept
      : __unit_count_{__unit_count}
  {}

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count() noexcept
  {
    return ::cuda::std::dynamic_extent;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr unsigned unit_count() const noexcept
  {
    return __unit_count_;
  }

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
      return _MappingResult::invalid();
    }

    const auto __prev_units_count = __prev_mapping_result.unit_count();
    const auto __prev_unit_rank   = __prev_mapping_result.unit_rank();

    _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__prev_units_count, __unit_count_),
                 "take mapping requires the previous mapping result to have at least _PrevMappingResult units");

    if (::cuda::std::cmp_greater_equal(__prev_unit_rank, __unit_count_))
    {
      return _MappingResult::invalid_with_group_count(__prev_mapping_result.group_count());
    }

    const auto __group_count = __prev_mapping_result.group_count();
    const auto __group_rank  = __prev_mapping_result.group_rank();
    const auto __unit_count  = __unit_count_;
    const auto __unit_rank   = __prev_unit_rank;
    const auto __lane_mask =
      (::cuda::std::is_same_v<_Unit, thread_level>)
        ? ::cuda::experimental::__make_lane_mask_for_n<_PrevMappingResult::is_always_contiguous()>(
            __prev_mapping_result.lane_mask(), __unit_count, __unit_rank)
        : __prev_mapping_result.lane_mask();
    return _MappingResult{__group_count, __group_rank, __unit_count, __unit_rank, __lane_mask};
  }
};

_CCCL_DEDUCTION_GUIDE_ATTRIBUTES take(unsigned) -> take<::cuda::std::dynamic_extent>;
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_TAKE_CUH
