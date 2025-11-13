// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/unsafe_bitcast.cuh>
#include <cub/thread/thread_operators.cuh> // is_cuda_minimum_maximum_v

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * Integer Utils
 **********************************************************************************************************************/

namespace detail
{
template <typename Input>
_CCCL_DEVICE _CCCL_FORCEINLINE auto split_integer(Input input)
{
  using namespace ::cuda::std;
  static_assert(__cccl_is_integer_v<Input>);
  constexpr auto half_bits = __num_bits_v<Input> / 2;
  using unsigned_t         = make_unsigned_t<Input>;
  using output_t           = __make_nbit_int_t<half_bits, is_signed_v<Input>>;
  auto input1              = static_cast<unsigned_t>(input);
  auto high                = static_cast<output_t>(input1 >> half_bits);
  auto low                 = static_cast<output_t>(input1);
  return array<output_t, 2>{high, low};
}

template <typename Input>
_CCCL_DEVICE _CCCL_FORCEINLINE auto merge_integers(Input inputA, Input inputB)
{
  using namespace ::cuda::std;
  static_assert(__cccl_is_integer_v<Input>);
  constexpr auto num_bits = __num_bits_v<Input>;
  using unsigned_t        = __make_nbit_uint_t<num_bits>;
  using unsigned_X2_t     = __make_nbit_uint_t<num_bits * 2>;
  using output_t          = __make_nbit_int_t<num_bits * 2, is_signed_v<Input>>;
  return static_cast<output_t>((static_cast<unsigned_X2_t>(inputA) << num_bits) | static_cast<unsigned_t>(inputB));
}

// When it is not possible to use native functionalities to compare floating-point values, we can convert them to
// an integer representation that preserves the order.
template <typename MinMaxOp, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE auto floating_point_to_comparable_int(MinMaxOp, T value)
{
  using namespace ::cuda::std;
  static_assert(::cuda::is_floating_point_v<T>);
  static_assert(is_cuda_minimum_maximum_v<MinMaxOp, T>);
  using signed_t        = __make_nbit_int_t<__num_bits_v<T>, true>;
  constexpr auto lowest = numeric_limits<signed_t>::lowest();
  constexpr auto is_max = is_cuda_maximum_v<MinMaxOp, T>;
  const auto nan        = is_max ? static_cast<T>(-numeric_limits<T>::quiet_NaN()) : numeric_limits<T>::quiet_NaN();
  auto value1           = ::cuda::std::isnan(value) ? nan : value;
  auto value_int        = cub::detail::unsafe_bitcast<signed_t>(value1);
  return static_cast<signed_t>(value_int < 0 ? lowest - value_int : value_int);
}

template <typename FloatingPointType, typename IntegerType>
_CCCL_DEVICE _CCCL_FORCEINLINE auto comparable_int_to_floating_point(IntegerType value)
{
  static_assert(::cuda::std::__cccl_is_integer_v<IntegerType>);
  constexpr auto lowest = ::cuda::std::numeric_limits<IntegerType>::lowest();
  auto value1           = static_cast<IntegerType>(value < 0 ? lowest - value : value);
  return cub::detail::unsafe_bitcast<FloatingPointType>(value1);
}
} // namespace detail
CUB_NAMESPACE_END
