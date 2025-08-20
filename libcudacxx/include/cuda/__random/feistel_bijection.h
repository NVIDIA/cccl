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

  uint64_t __right_side_bits{};
  uint64_t __left_side_bits{};
  uint64_t __right_side_mask{};
  uint64_t __left_side_mask{};
  uint32_t __keys[__num_rounds] = {};

  struct __decomposed
  {
    uint32_t __low;
    uint32_t __high;
  };

public:
  using index_type = uint64_t;

  _CCCL_HIDE_FROM_ABI constexpr __feistel_bijection() noexcept = default;

  template <class _RNG>
  _CCCL_API __feistel_bijection(uint64_t __num_elements, _RNG&& __gen)
  {
    const uint64_t __total_bits = (::cuda::std::max) (uint64_t{4}, ::cuda::std::bit_ceil(__num_elements));

    // Half bits rounded down
    __left_side_bits = __total_bits / 2;
    __left_side_mask = (1ull << __left_side_bits) - 1;
    // Half the bits rounded up
    __right_side_bits = __total_bits - __left_side_bits;
    __right_side_mask = (1ull << __right_side_bits) - 1;

    ::cuda::std::uniform_int_distribution<uint32_t> dist{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (uint32_t i = 0; i < __num_rounds; i++)
    {
      __keys[i] = dist(__gen);
    }
  }

  [[nodiscard]] _CCCL_API constexpr uint64_t size() const noexcept
  {
    return 1ull << (__left_side_bits + __right_side_bits);
  }

  [[nodiscard]] _CCCL_API constexpr uint64_t operator()(const uint64_t __val) const noexcept
  {
    __decomposed __state = {static_cast<uint32_t>(__val >> __right_side_bits),
                            static_cast<uint32_t>(__val & __right_side_mask)};
    for (uint32_t i = 0; i < __num_rounds; i++)
    {
      constexpr uint64_t __m0  = 0xD2B74407B1CE6E93;
      const uint64_t __product = __m0 * __state.__high;
      const uint32_t __high    = static_cast<uint32_t>(__product >> 32);
      uint32_t __low           = static_cast<uint32_t>(__product);
      __low                    = (__low << (__right_side_bits - __left_side_bits)) | __state.__low >> __left_side_bits;
      __state.__high           = ((__high ^ __keys[i]) ^ __state.__low) & __left_side_mask;
      __state.__low            = __low & __right_side_mask;
    }
    // Combine the left and right sides together to get result
    return (static_cast<uint64_t>(__state.__high) << __right_side_bits) | static_cast<uint64_t>(__state.__low);
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___RANDOM_FEISTEL_BIJECTION_H
