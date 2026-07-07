//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_MAPPING_MAPPING_RESULT_CUH
#define _CUDA_EXPERIMENTAL___GROUP_MAPPING_MAPPING_RESULT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__warp/lane_mask.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>

#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental
{
template <::cuda::std::size_t _StaticGroupCount, ::cuda::std::size_t _StaticCount, bool _IsExhaustive, bool _IsContiguous>
struct __mapping_result
{
  unsigned __group_count_;
  unsigned __group_rank_;
  unsigned __unit_count_;
  unsigned __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result invalid() noexcept
  {
    return {__invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result
  invalid_with_group_count(unsigned __group_count) noexcept
  {
    return {__group_count,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count() noexcept
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned group_count() const noexcept
  {
    if constexpr (_StaticGroupCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<unsigned>(_StaticGroupCount);
    }
    else
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(__group_count_ != __invalid_count_or_rank,
                     "getting group count by a unit that was not part of the parent group is not allowed");
      }
      return __group_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned group_rank() const noexcept
  {
    if constexpr (!_IsExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
    }
    return __group_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count() noexcept
  {
    return _StaticCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned unit_count() const noexcept
  {
    if constexpr (_StaticCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<unsigned>(_StaticCount);
    }
    else
    {
      if constexpr (!_IsExhaustive)
      {
        _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
      }
      return __unit_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned unit_rank() const noexcept
  {
    if constexpr (!_IsExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit rank of thread that is not part of the group is UB");
    }
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const noexcept
  {
    if constexpr (!_IsExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting lane mask of thread that is not part of the group is UB");
    }
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
  {
    if constexpr (_IsExhaustive)
    {
      return true;
    }
    else
    {
      return __unit_rank_ != __invalid_count_or_rank;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive() noexcept
  {
    return _IsExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous() noexcept
  {
    return _IsContiguous;
  }
};

template <bool _IsContiguous>
[[nodiscard]] _CCCL_DEVICE_API inline ::cuda::device::lane_mask
__make_lane_mask_for_n(::cuda::device::lane_mask __prev_lane_mask, unsigned __n, unsigned __rank) noexcept
{
  if constexpr (_IsContiguous)
  {
    auto __lane_mask  = __prev_lane_mask;
    const auto __lane = ::cuda::ptx::get_sreg_laneid();

    if (__lane > __rank)
    {
      __lane_mask &= ::cuda::device::lane_mask::all() << (__lane - __rank);
    }
    if (__lane + (__n - __rank) < 32)
    {
      __lane_mask &= ::cuda::device::lane_mask::all() >> (32 - __lane - (__n - __rank));
    }
    return __lane_mask;
  }
  else
  {
    auto __lane_mask = ::cuda::device::lane_mask::this_lane();

    const auto __less_mask = __prev_lane_mask & ::cuda::device::lane_mask::all_less();
    const auto __nless     = ::cuda::std::popcount(__less_mask.value());
    if (__nless > __rank)
    {
      const auto __nless_to_remove = __nless - __rank;
      const auto __last_to_remove  = ::__fns(__less_mask.value(), 0, __nless_to_remove);
      __lane_mask |= ::cuda::device::lane_mask{__less_mask.value() & (~0u << (__last_to_remove + 1))};
    }
    else
    {
      __lane_mask |= __less_mask;
    }

    const auto __greater_mask = __prev_lane_mask & ::cuda::device::lane_mask::all_greater();
    const auto __ngreater     = ::cuda::std::popcount(__greater_mask.value());
    if (__rank + __ngreater >= __n)
    {
      const auto __ngreater_to_keep = __n - __rank;
      const auto __first_to_remove  = ::__fns(__greater_mask.value(), 0, __ngreater_to_keep);
      __lane_mask |= ::cuda::device::lane_mask{__greater_mask.value() & ((1u << __first_to_remove) - 1u)};
    }
    else
    {
      __lane_mask |= __greater_mask;
    }
    return __lane_mask;
  }
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_MAPPING_RESULT_CUH
