// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename Input, ::cuda::std::size_t NumWords = sizeof(Input) / sizeof(unsigned)>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE ::cuda::std::array<unsigned, NumWords> to_words(const Input input)
{
  static_assert(::cuda::std::is_integral_v<Input>);
  static_assert(sizeof(Input) == 2 * sizeof(unsigned) || sizeof(Input) == 4 * sizeof(unsigned));
  using unsigned_t = ::cuda::std::make_unsigned_t<Input>;

  constexpr auto word_bits  = 32;
  const auto unsigned_input = static_cast<unsigned_t>(input);
  ::cuda::std::array<unsigned, NumWords> result{};
  for (::cuda::std::size_t word = 0; word < NumWords; ++word)
  {
    result[word] = static_cast<unsigned>(unsigned_input >> (word * word_bits));
  }
  return result;
}

template <typename Output, ::cuda::std::size_t Size = sizeof(Output) / sizeof(unsigned)>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE Output from_words(const ::cuda::std::array<unsigned, Size>& input)
{
  static_assert(::cuda::std::is_integral_v<Output>);
  static_assert(sizeof(Output) == Size * sizeof(unsigned));
  static_assert(Size == 2 || Size == 4);

  using unsigned_t         = ::cuda::std::make_unsigned_t<Output>;
  constexpr auto word_bits = 32;
  unsigned_t result{};
  for (::cuda::std::size_t word = 0; word < Size; ++word)
  {
    result |= static_cast<unsigned_t>(input[word]) << (word * word_bits);
  }
  return static_cast<Output>(result);
}

// reconstructs a 32-bit word from its reduced partial sums and updates the carry
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned
reconstruct_word_with_carry(const unsigned low_sum, const unsigned high_sum, unsigned& carry)
{
  const auto sum_lo = low_sum + carry;
  // the following code is equivalent to:
  //   const auto sum_hi = high_sum + (sum_lo >> 27);
  //   carry             = sum_hi >> 5;
  //   return (sum_lo & 0x07FF'FFFFu) | ((sum_hi & 0b11111u) << 27);
  const auto sum_hi = (high_sum << 3) + (sum_lo >> 24);
  carry             = sum_hi >> 8;
  return __byte_perm(sum_lo, sum_hi, 0x4210);
}
} // namespace detail

CUB_NAMESPACE_END
