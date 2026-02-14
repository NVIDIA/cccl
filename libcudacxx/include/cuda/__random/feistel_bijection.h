//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___RANDOM_FEISTEL_BIJECTION_H
#define _CUDA___RANDOM_FEISTEL_BIJECTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__random/uniform_int_distribution.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief A Feistel cipher for operating on power of two sized problems
class __feistel_bijection
{
private:
  static constexpr uint32_t __num_rounds = 24;

  uint64_t __R_bits_{};
  uint64_t __L_bits_{};
  uint64_t __R_mask_{};
  uint64_t __L_mask_{};
  uint32_t __keys_[__num_rounds] = {};

public:
  using index_type = uint64_t;

  _CCCL_HIDE_FROM_ABI constexpr __feistel_bijection() noexcept = default;

  template <class _RNG>
  _CCCL_API __feistel_bijection(uint64_t __num_elements, _RNG&& __gen)
  {
    // Calculate number of bits needed to represent num_elements - 1
    // Prevent zero
    const uint64_t __max_index  = ::cuda::std::max(static_cast<uint64_t>(1), __num_elements) - 1;
    const uint64_t __total_bits = static_cast<uint64_t>(::cuda::std::max(8, ::cuda::std::bit_width(__max_index)));

    // Half bits rounded down
    __L_bits_ = __total_bits / 2;
    __L_mask_ = (1ull << __L_bits_) - 1;
    // Half the bits rounded up
    __R_bits_ = __total_bits - __L_bits_;
    __R_mask_ = (1ull << __R_bits_) - 1;

    ::cuda::std::uniform_int_distribution<uint32_t> __dist{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (uint32_t i = 0; i < __num_rounds; i++)
    {
      __keys_[i] = __dist(__gen);
    }
  }

  [[nodiscard]] _CCCL_API constexpr uint64_t size() const noexcept
  {
    return 1ull << (__L_bits_ + __R_bits_);
  }

  [[nodiscard]] _CCCL_API constexpr uint64_t operator()(const uint64_t __val) const noexcept
  {
    // Mitchell, Rory, et al. "Bandwidth-optimal random shuffling for GPUs." ACM Transactions on Parallel Computing 9.1
    // (2022): 1-20.
    uint32_t __L = static_cast<uint32_t>(__val >> __R_bits_);
    uint32_t __R = static_cast<uint32_t>(__val & __R_mask_);
    for (uint32_t __i = 0; __i < __num_rounds; __i++)
    {
      constexpr uint64_t __m0  = 0xD2B74407B1CE6E93;
      const uint64_t __product = __m0 * __L;
      uint32_t __F_k           = (__product >> 32) ^ __keys_[__i];
      uint32_t __B_k           = static_cast<uint32_t>(__product);
      uint32_t __L_prime       = __F_k ^ __R;

      uint32_t __R_prime = (__B_k << (__R_bits_ - __L_bits_)) | __R >> __L_bits_;
      __L                = __L_prime & __L_mask_;
      __R                = __R_prime & __R_mask_;
    }
    // Combine the left and right sides together to get result
    return (static_cast<uint64_t>(__L) << __R_bits_) | static_cast<uint64_t>(__R);
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___RANDOM_FEISTEL_BIJECTION_H
