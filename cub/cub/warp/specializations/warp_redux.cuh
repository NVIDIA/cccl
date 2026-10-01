// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Helpers for warp-level REDUX reductions.
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/integer_utility.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/std/__floating_point/cast.h> // IWYU pragma: keep
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
//----------------------------------------------------------------------------------------------------------------------
// Redux Traits

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_op_supported_sm80 =
  ::cuda::std::is_integral_v<T> && sizeof(T) <= sizeof(unsigned)
  && (is_cuda_minimum_maximum_v<ReduceOp, T> || is_cuda_std_plus_v<ReduceOp, T> || is_cuda_std_bitwise_v<ReduceOp, T>);

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_bitwise_large_supported =
  ::cuda::std::is_unsigned_v<T> && sizeof(T) > sizeof(unsigned) && is_cuda_std_bitwise_v<ReduceOp, T>;

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_plus_64bit_supported =
  ::cuda::std::is_integral_v<T> && (sizeof(T) == sizeof(unsigned) * 2) && is_cuda_std_plus_v<ReduceOp, T>;

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_plus_128bit_supported =
  ::cuda::std::is_integral_v<T> && (sizeof(T) == sizeof(unsigned) * 4) && is_cuda_std_plus_v<ReduceOp, T>;

template <typename Op, typename T, typename ReduceOp = ::cuda::std::remove_cvref_t<Op>>
inline constexpr bool is_warp_redux_min_max_f32_supported =
  __cccl_ptx_isa >= 860 && (::cuda::std::is_same_v<T, float> || is_half_v<T> || is_bfloat16_v<T>)
  && is_cuda_minimum_maximum_v<ReduceOp, T>;

template <typename Op, typename T>
inline constexpr bool is_warp_redux_op_supported =
  is_warp_redux_op_supported_sm80<Op, T> //
  || is_warp_redux_plus_64bit_supported<Op, T> //
  || is_warp_redux_plus_128bit_supported<Op, T> //
  || is_warp_redux_bitwise_large_supported<Op, T> //
  || is_warp_redux_min_max_f32_supported<Op, T>;

//----------------------------------------------------------------------------------------------------------------------
// SM80 Redux

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_sm80(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_op_supported_sm80<ReductionOp, T>, "Reduction operator not supported");
  _CCCL_ASSERT(mask != 0, "Mask must not be 0");

  using promotion_t = ::cuda::std::conditional_t<::cuda::std::is_signed_v<T>, int, unsigned>;
  const auto value  = static_cast<promotion_t>(input);
  if constexpr (is_cuda_maximum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_max_sync(mask, value));
  }
  else if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_min_sync(mask, value));
  }
  else if constexpr (is_cuda_std_plus_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_add_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_and_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_and_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_or_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_or_sync(mask, value));
  }
  else if constexpr (is_cuda_std_bit_xor_v<ReductionOp, T>)
  {
    return static_cast<T>(__reduce_xor_sync(mask, value));
  }
  else
  {
    _CCCL_UNREACHABLE();
    return T{};
  }
}

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_plus_64bit(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_plus_64bit_supported<ReductionOp, T>, "Reduction operator not supported");
  const auto [low, high] = CUB_NS_QUALIFIER::detail::to_words(input);
  // The following algorithms implements (multi-precision) column addition with deferred carry propagation, see
  // https://eprint.iacr.org/2019/794.pdf#page=10 for more details.
  // the core idea is to split the inputs in columns (bit ranges) where the overflow is not possible.
  // Base 10 example:
  //    78
  // +  67
  // +  59
  // -----
  //   204
  //
  // For the binary case, we split each value in three:
  //  - 32-bit high: 32-63 bits
  //  - 8-bit  low1: 24-31 bits
  //  - 24-bit low0: 0-24 bits
  // we can compute the reduction of each column independently, avoiding overflows, and then combine the results.

  const auto low1 = low >> 24; // high 8 bits
  const auto low0 = low & 0x00FFFFFFu; // low 24 bits

  // the following reductions cannot overflow
  const auto low0_sum = __reduce_add_sync(mask, low0);
  const auto low1_sum = __reduce_add_sync(mask, low1);
  const auto low_sum  = low1_sum + (low0_sum >> 24); // the carry of low0_sum is added to low1_sum (24-31 bits)

  // concatenate the first 24 bits of low0_sum (0x210) with the last 8 bits of carry (0x4000)
  const auto ret_lo = __byte_perm(low0_sum, low_sum, 0x4210);
  const auto ret_hi = __reduce_add_sync(mask, high) + (low_sum >> 8); // low_sum >> 8 is the carry of the low bits

  return CUB_NS_QUALIFIER::detail::from_words<T>(::cuda::std::array<::cuda::std::uint32_t, 2>{ret_lo, ret_hi});
}

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_plus_128bit(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_plus_128bit_supported<ReductionOp, T>, "Reduction operator not supported");
  const auto [word0, word1, word2, word3] = CUB_NS_QUALIFIER::detail::to_words(input);
  // In a similar way of the 64-bit case, we split the input in:
  // 27-bit of word0: 0-26 bits
  // 27-bit of word1: 32-58 bits
  // 27-bit of word2: 64-90 bits
  // each reduction cannot overflow
  const auto low0 = __reduce_add_sync(mask, word0 & 0x07FF'FFFFu);
  const auto low1 = __reduce_add_sync(mask, word1 & 0x07FF'FFFFu);
  const auto low2 = __reduce_add_sync(mask, word2 & 0x07FF'FFFFu);

  // we extract the high 5 bits of each word
  // 5 bits max value=31, 31 * 32 lanes=992, that can be represented with 10 bits
  // we perform the reduction of all three values in a single operation of __reduce_add_sync
  //
  // word0_high = word0 >> 27;
  // word1_high = word1 >> 27;
  // word2_high = word2 >> 27;
  // packed = word0_high | (word1_high << 10) | (word2_high << 20);
  //
  // the previous code can be implemented slightly more efficiently with __funnelshift_l
  // __funnelshift_l(lo, hi, S) == (hi << S) | (lo >> (32 - S))
  const auto packed_a  = word2 >> 22; // top 10 bits of word2
  const auto packed_b  = __funnelshift_l(word1, packed_a, 10); // concat(word2 top 10 bits, word1 low 22 bits)
  const auto packed_c  = __funnelshift_l(word0, packed_b, 5); // concat(packed_b top 5 bits, word0 low 27 bits)
  const auto packed    = packed_c & 0b00000'11111'00000'11111'00000'11111u; // select only the 5-bit fields
  const auto high_sums = __reduce_add_sync(mask, packed);

  const auto high_sum0 = high_sums & 0b11111'11111u; // extract first 10 bits
  const auto high_sum1 = (high_sums >> 10) & 0b11111'11111u; // extract next 10 bits
  const auto high_sum2 = high_sums >> 20; // extract last 10 bits
  uint32_t carry       = 0;
  const auto ret0      = CUB_NS_QUALIFIER::detail::reconstruct_word_with_carry(low0, high_sum0, carry);
  const auto ret1      = CUB_NS_QUALIFIER::detail::reconstruct_word_with_carry(low1, high_sum1, carry);
  const auto ret2      = CUB_NS_QUALIFIER::detail::reconstruct_word_with_carry(low2, high_sum2, carry);
  const auto ret3      = __reduce_add_sync(mask, word3) + carry;

  return CUB_NS_QUALIFIER::detail::from_words<T>({ret0, ret1, ret2, ret3});
}

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
warp_redux_bitwise_large(const T input, const ::cuda::std::uint32_t mask, ReductionOp reduction_op)
{
  static_assert(is_warp_redux_bitwise_large_supported<ReductionOp, T>, "Reduction operator not supported");
  constexpr int chunk_bits  = 32; // 32 bits
  constexpr int num_chunks  = sizeof(T) / sizeof(unsigned);
  const auto generalized_op = CUB_NS_QUALIFIER::detail::generalize_operator(reduction_op); // map bit_and<uint64_t> to
                                                                                           // bit_and<>
  // do not use bit_cast/memcpy to avoid potential performance issues
  T output{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int chunk = 0; chunk < num_chunks; ++chunk)
  {
    const int shift    = chunk * chunk_bits;
    const auto value   = static_cast<unsigned>(input >> shift);
    const auto reduced = CUB_NS_QUALIFIER::detail::warp_redux_sm80(value, mask, generalized_op);
    output |= static_cast<T>(reduced) << shift;
  }
  return output;
}

//----------------------------------------------------------------------------------------------------------------------
// Floating-point Min/Max Redux

#if __cccl_ptx_isa >= 860

#  define _CUB_REDUX_FLOAT_OP(_CCCL_PTX_OP)                                                        \
    [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE float redux_min_max_f32_##_CCCL_PTX_OP##_ptx( \
      const float value, ::cuda::std::uint32_t mask)                                               \
    {                                                                                              \
      float result;                                                                                \
      asm volatile("{"                                                                             \
                   "redux.sync." #_CCCL_PTX_OP ".f32 %0, %1, %2;"                                  \
                   "}"                                                                             \
                   : "=f"(result)                                                                  \
                   : "f"(value), "r"(mask));                                                       \
      return result;                                                                               \
    }

_CUB_REDUX_FLOAT_OP(min)
_CUB_REDUX_FLOAT_OP(max)

// TODO(fbusato): min_abs, max_abs are also available but we need to introduce the corresposing operators
// _CUB_REDUX_FLOAT_OP(min_abs)
// _CUB_REDUX_FLOAT_OP(max_abs)

#  undef _CUB_REDUX_FLOAT_OP

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API
_CCCL_FORCEINLINE T warp_redux_min_max_f32(const T input, const ::cuda::std::uint32_t mask, ReductionOp)
{
  static_assert(is_warp_redux_min_max_f32_supported<ReductionOp, T>, "Reduction operator not supported");
  _CCCL_ASSERT(mask != 0, "Mask must not be 0");

  const float value = ::cuda::std::__fp_cast<float>(input);
  float result;
  if constexpr (is_cuda_minimum_v<ReductionOp, T>)
  {
    result = CUB_NS_QUALIFIER::detail::redux_min_max_f32_min_ptx(value, mask);
  }
  else
  {
    result = CUB_NS_QUALIFIER::detail::redux_min_max_f32_max_ptx(value, mask);
  }
  return ::cuda::std::__fp_cast<T>(result);
}

#endif // __cccl_ptx_isa >= 860

//----------------------------------------------------------------------------------------------------------------------
// Redux Dispatch

template <typename T, typename ReductionOp>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr ::cuda::std::optional<T>
warp_redux(const T input, const ::cuda::std::uint32_t mask, ReductionOp reduction_op)
{
  static_assert(is_warp_redux_op_supported<ReductionOp, T>, "Reduction operator not supported");
  if constexpr (is_warp_redux_op_supported_sm80<ReductionOp, T>)
  { // NOLINT(bugprone-branch-clone)
    NV_IF_TARGET(NV_PROVIDES_SM_80, (return CUB_NS_QUALIFIER::detail::warp_redux_sm80(input, mask, reduction_op);))
  }
  else if constexpr (is_warp_redux_plus_64bit_supported<ReductionOp, T>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return CUB_NS_QUALIFIER::detail::warp_redux_plus_64bit(input, mask, reduction_op);))
  }
  else if constexpr (is_warp_redux_plus_128bit_supported<ReductionOp, T>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return CUB_NS_QUALIFIER::detail::warp_redux_plus_128bit(input, mask, reduction_op);))
  }
  else if constexpr (is_warp_redux_bitwise_large_supported<ReductionOp, T>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_80,
                 (return CUB_NS_QUALIFIER::detail::warp_redux_bitwise_large(input, mask, reduction_op);))
  }
  else if constexpr (is_warp_redux_min_max_f32_supported<ReductionOp, T>)
  {
    // Before PTX ISA 8.8, float reductions are only supported on sm100a.
#if __cccl_ptx_isa >= 880
    NV_IF_TARGET(NV_HAS_FEATURE_SM_100f,
                 (return CUB_NS_QUALIFIER::detail::warp_redux_min_max_f32(input, mask, reduction_op);))
#else // ^^^ __cccl_ptx_isa >= 880 ^^^ / vvv __cccl_ptx_isa < 880 vvv
    NV_IF_TARGET(NV_HAS_FEATURE_SM_100a,
                 (return CUB_NS_QUALIFIER::detail::warp_redux_min_max_f32(input, mask, reduction_op);))
#endif // ^^^ __cccl_ptx_isa < 880 ^^^
  }
  return ::cuda::std::nullopt;
}
} // namespace detail

CUB_NAMESPACE_END
