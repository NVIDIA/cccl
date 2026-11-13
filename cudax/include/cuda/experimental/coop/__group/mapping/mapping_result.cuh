//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH
#define _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__bit/bit_fns.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__warp/lane_mask.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/coop/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

// todo(dabayer): do we want to always use uint32_t for all counts/ranks?

namespace cuda::experimental::coop
{
template <::cuda::std::size_t _StaticGroupCount, ::cuda::std::size_t _StaticUnitCount>
struct __uniform_static_unit_count;

template <::cuda::std::size_t... _StaticUnitCounts>
struct __static_unit_counts;

template <::cuda::std::size_t _StaticGroupCount, ::cuda::std::size_t _GroupRank, ::cuda::std::size_t _StaticUnitCount>
struct __per_group_count;

template <class _CountsTag, bool _IsAlwaysExhaustive, bool _IsAlwaysContiguous>
struct __mapping_result;

template <::cuda::std::size_t _StaticGroupCount,
          ::cuda::std::size_t _StaticUnitCount,
          bool _IsAlwaysExhaustive,
          bool _IsAlwaysContiguous>
struct __mapping_result<__uniform_static_unit_count<_StaticGroupCount, _StaticUnitCount>,
                        _IsAlwaysExhaustive,
                        _IsAlwaysContiguous>
{
  ::cuda::std::uint32_t __group_count_;
  ::cuda::std::uint32_t __group_rank_;
  ::cuda::std::uint32_t __unit_count_;
  ::cuda::std::uint32_t __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid()
  {
    return {__invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            __invalid_count_or_rank,
            ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group count of thread that is not part of the group is UB");
    }

    if constexpr (_StaticGroupCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
    }
    else
    {
      return __group_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
    }

    if constexpr (_StaticGroupCount == 1)
    {
      return 0;
    }
    else
    {
      return __group_rank_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    if constexpr (_StaticGroupCount != ::cuda::std::dynamic_extent)
    {
      _CCCL_ASSERT(__i < _StaticGroupCount, "__i out of range");
    }
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit count of thread that is not part of the group is UB");
    }

    if constexpr (_StaticUnitCount != ::cuda::std::dynamic_extent)
    {
      return static_cast<::cuda::std::uint32_t>(_StaticUnitCount);
    }
    else
    {
      return __unit_count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit rank of thread that is not part of the group is UB");
    }

    if constexpr (_StaticUnitCount == 1)
    {
      return 0;
    }
    else
    {
      return __unit_rank_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting lane mask of thread that is not part of the group is UB");
    }
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const
  {
    if constexpr (_IsAlwaysExhaustive)
    {
      return true;
    }
    else
    {
      return __unit_rank_ != __invalid_count_or_rank;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _IsAlwaysContiguous;
  }
};

template <::cuda::std::size_t... _StaticUnitCounts, bool _IsAlwaysExhaustive, bool _IsAlwaysContiguous>
struct __mapping_result<__static_unit_counts<_StaticUnitCounts...>, _IsAlwaysExhaustive, _IsAlwaysContiguous>
{
  static constexpr ::cuda::std::size_t _StaticGroupCount = sizeof...(_StaticUnitCounts);

  static_assert(_StaticGroupCount > 0, "per-group unit counts require at least one group");
  static_assert(((_StaticUnitCounts != ::cuda::std::dynamic_extent) && ...),
                "per-group unit counts must be statically known");

  ::cuda::std::uint32_t __group_rank_;
  ::cuda::std::uint32_t __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result __invalid()
  {
    return {__invalid_count_or_rank, __invalid_count_or_rank, ::cuda::device::lane_mask::none()};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t group_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group count of thread that is not part of the group is UB");
    }
    return static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t group_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting group rank of thread that is not part of the group is UB");
    }
    return __group_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return ((static_unit_count(0) == _StaticUnitCounts) && ...);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    if constexpr (has_uniform_static_unit_count())
    {
      return static_unit_count(0);
    }
    else
    {
      return ::cuda::std::dynamic_extent;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    _CCCL_ASSERT(__i < _StaticGroupCount, "__i out of range");

    // Use fold expression rather than constexpr array to keep runtime queries in registers.
    ::cuda::std::size_t __index  = 0;
    ::cuda::std::size_t __result = 0;
    ((__result = (__i == __index++) ? _StaticUnitCounts : __result), ...);
    return __result;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_count() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit count of thread that is not part of the group is UB");
    }
    return static_cast<::cuda::std::uint32_t>(static_unit_count(__group_rank_));
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting unit rank of thread that is not part of the group is UB");
    }
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    if constexpr (!_IsAlwaysExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting lane mask of thread that is not part of the group is UB");
    }
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const
  {
    if constexpr (_IsAlwaysExhaustive)
    {
      return true;
    }
    else
    {
      return __unit_rank_ != __invalid_count_or_rank;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _IsAlwaysContiguous;
  }
};

// This mapping result is only ever used in group.with_static_unit_count(...) where it's constructed from the group's
// mapping result. That means that the mapping result is always valid. We still keep the is_always_exhaustive() property
// from the original mapping result.
template <::cuda::std::size_t _StaticGroupCount,
          ::cuda::std::size_t _GroupRank,
          ::cuda::std::size_t _StaticUnitCount,
          bool _IsAlwaysExhaustive,
          bool _IsAlwaysContiguous>
struct __mapping_result<__per_group_count<_StaticGroupCount, _GroupRank, _StaticUnitCount>,
                        _IsAlwaysExhaustive,
                        _IsAlwaysContiguous>
{
  static_assert(_StaticGroupCount != ::cuda::std::dynamic_extent, "group count must be statically known");
  static_assert(_GroupRank < _StaticGroupCount, "group rank out of range");
  static_assert(_StaticUnitCount != ::cuda::std::dynamic_extent, "unit count must be statically known");

  ::cuda::std::uint32_t __unit_rank_;
  ::cuda::device::lane_mask __lane_mask_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_group_count()
  {
    return _StaticGroupCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t group_count() const
  {
    return static_cast<::cuda::std::uint32_t>(_StaticGroupCount);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t group_rank() const
  {
    return static_cast<::cuda::std::uint32_t>(_GroupRank);
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool has_uniform_static_unit_count()
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count()
  {
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_unit_count(::cuda::std::size_t __i)
  {
    _CCCL_ASSERT(__i < _StaticGroupCount, "__i out of range");
    return _StaticUnitCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint32_t unit_count() const
  {
    return static_cast<::cuda::std::uint32_t>(_StaticUnitCount);
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::std::uint32_t unit_rank() const
  {
    return __unit_rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask lane_mask() const
  {
    return __lane_mask_;
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool is_valid() const
  {
    return true;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_exhaustive()
  {
    return _IsAlwaysExhaustive;
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr bool is_always_contiguous()
  {
    return _IsAlwaysContiguous;
  }
};

template <bool _IsAlwaysContiguous>
[[nodiscard]] _CCCL_DEVICE_API ::cuda::device::lane_mask __make_lane_mask_for_n(
  ::cuda::device::lane_mask __prev_lane_mask, ::cuda::std::uint32_t __n, ::cuda::std::uint32_t __rank)
{
  if constexpr (_IsAlwaysContiguous)
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
      const auto __last_to_remove = ::cuda::bit_fns(__less_mask.value(), __nless - __rank - 1);
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
      const auto __first_to_remove = ::cuda::bit_fns(__greater_mask.value(), __n - __rank - 1);
      __lane_mask |= ::cuda::device::lane_mask{__greater_mask.value() & ((1u << __first_to_remove) - 1u)};
    }
    else
    {
      __lane_mask |= __greater_mask;
    }
    return __lane_mask;
  }
}
} // namespace cuda::experimental::coop

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_COOP___GROUP_MAPPING_MAPPING_RESULT_CUH
