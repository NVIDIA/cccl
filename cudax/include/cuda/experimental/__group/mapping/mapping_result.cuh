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
  unsigned __count_;
  unsigned __rank_;

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result invalid() noexcept
  {
    return {__invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank};
  }

  [[nodiscard]] _CCCL_DEVICE_API static constexpr __mapping_result
  invalid_with_group_count(unsigned __group_count) noexcept
  {
    return {__group_count, __invalid_count_or_rank, __invalid_count_or_rank, __invalid_count_or_rank};
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

  [[nodiscard]] _CCCL_DEVICE_API static constexpr ::cuda::std::size_t static_count() noexcept
  {
    return _StaticCount;
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned count() const noexcept
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
      return __count_;
    }
  }

  [[nodiscard]] _CCCL_DEVICE_API unsigned rank() const noexcept
  {
    if constexpr (!_IsExhaustive)
    {
      _CCCL_ASSERT(is_valid(), "getting rank of thread that is not part of the group is UB");
    }
    return __rank_;
  }

  [[nodiscard]] _CCCL_DEVICE_API bool is_valid() const noexcept
  {
    if constexpr (_IsExhaustive)
    {
      return true;
    }
    else
    {
      return __rank_ != __invalid_count_or_rank;
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
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_MAPPING_MAPPING_RESULT_CUH
