//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BIT_BIT_FNS_H
#define _CUDA___BIT_BIT_FNS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/integral.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr int __bit_fns_impl(const _Tp __v, const int __rank) noexcept
{
  using ::cuda::std::uint32_t;

  if constexpr (sizeof(_Tp) <= sizeof(uint32_t))
  {
    constexpr auto __nbits = ::cuda::std::__num_bits_v<_Tp>;

    // A zero mask preserves the -1 sentinel when the requested rank is absent.
    uint32_t __mask = (__rank < ::cuda::std::popcount(__v)) ? static_cast<_Tp>(~_Tp{0}) : 0u;

    // Find the shortest low-bit prefix containing the selected bit, keeping its rank fixed.
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int __step = __nbits / 2; __step != 0; __step /= 2)
    {
      const auto __candidate = __mask >> __step;
      __mask                 = (::cuda::std::popcount(__v & __candidate) > __rank) ? __candidate : __mask;
    }
    return static_cast<int>(::cuda::std::__bit_log2(__mask));
  }
  else
  {
    // Discard half the bits at each step until the search fits in a 32-bit word.
    constexpr int __half_bits = ::cuda::std::__num_bits_v<_Tp> / 2;
    using _Half               = ::cuda::std::__make_nbit_uint_t<__half_bits>;
    const auto __lo           = static_cast<_Half>(__v);
    const int __lo_count      = ::cuda::std::popcount(__lo);
    const bool __is_in_hi     = __rank >= __lo_count;
    const auto __word         = __is_in_hi ? static_cast<_Half>(__v >> __half_bits) : __lo;
    const int __word_rank     = __is_in_hi ? __rank - __lo_count : __rank;
    const int __position      = ::cuda::__bit_fns_impl(__word, __word_rank);
    // The half's offset occupies a bit above every valid position in that half.
    // Bitwise OR adds the offset while preserving the all-ones -1 sentinel.
    return __position | (__is_in_hi ? __half_bits : 0);
  }
}

//! @brief Finds the position of the set bit with rank \p __rank in \p __value, counting set bits from the least
//! significant one.
//!
//! @param[in] __value The unsigned integer value to search.
//! @param[in] __rank The zero-based rank of the set bit to find.
//! @return The zero-based position of the selected set bit, or `-1` if \p __value has fewer than `__rank + 1` set bits.
//! @pre `0 <= __rank && __rank < num_bits(_Tp)`.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr int bit_fns(const _Tp __value, const int __rank) noexcept
{
  constexpr auto __nbits = ::cuda::std::__num_bits_v<_Tp>;
  _CCCL_ASSERT(__rank >= 0 && __rank < __nbits, "rank out of range");
  int __ret{};
  if (_CCCL_IS_CONSTANT_P(__rank) && __rank == __nbits - 1)
  {
    __ret = (__value == static_cast<_Tp>(-1)) ? __nbits - 1 : -1;
  }
  else
  {
    __ret = ::cuda::__bit_fns_impl(__value, __rank);
  }
  _CCCL_ASSUME(__ret >= -1 && __ret < __nbits);
  return __ret;
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___BIT_BIT_FNS_H
