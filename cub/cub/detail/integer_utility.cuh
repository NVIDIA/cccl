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

#include <cub/detail/type_traits.cuh>

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_signed.h>
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

template <typename T>
inline constexpr bool is_floating_point_comparable_v =
  is_half_v<T> || is_bfloat16_v<T> || ::cuda::std::is_same_v<T, float> || ::cuda::std::is_same_v<T, double>
#if _CCCL_HAS_FLOAT128()
  || ::cuda::std::is_same_v<T, __float128>
#endif // _CCCL_HAS_FLOAT128()
  ;

// Maps floating-point values to signed integers of the same size, preserving their order
template <typename T>
[[nodiscard]] _CCCL_API constexpr auto floating_point_to_comparable_int(const T value) noexcept
{
  static_assert(is_floating_point_comparable_v<T>, "T must be __half, __nv_bfloat16, float, double, or __float128");
  using signed_t = ::cuda::std::make_signed_t<::cuda::std::__fp_storage_of_t<T>>;

  const auto bits = static_cast<signed_t>(::cuda::std::__fp_get_storage(value));
  return bits < 0 ? static_cast<signed_t>(bits ^ ::cuda::std::numeric_limits<signed_t>::max()) : bits;
}

// Inverse mapping from (comparable) signed integers to floating-point values
template <typename T, typename Int>
[[nodiscard]] _CCCL_API constexpr T comparable_int_to_floating_point(const Int value) noexcept
{
  static_assert(is_floating_point_comparable_v<T>, "T must be __half, __nv_bfloat16, float, double, or __float128");
  using storage_t = ::cuda::std::__fp_storage_of_t<T>;
  static_assert(::cuda::std::is_same_v<Int, ::cuda::std::make_signed_t<storage_t>>, "Int must match the storage of T");

  const auto bits = value < 0 ? static_cast<Int>(value ^ ::cuda::std::numeric_limits<Int>::max()) : value;
  return ::cuda::std::__fp_from_storage<T>(static_cast<storage_t>(bits));
}
} // namespace detail

CUB_NAMESPACE_END
