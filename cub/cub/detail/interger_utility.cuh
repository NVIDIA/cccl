// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename Input>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE auto split_integer(const Input input)
{
  static_assert(::cuda::std::is_integral_v<Input>);
  constexpr auto half_bits = ::cuda::std::__num_bits_v<Input> / 2;
  using unsigned_t         = ::cuda::std::make_unsigned_t<Input>;
  using output_t           = ::cuda::std::__make_nbit_int_t<half_bits, ::cuda::std::is_signed_v<Input>>;
  const auto input1        = static_cast<unsigned_t>(input);
  const auto high          = static_cast<output_t>(input1 >> half_bits);
  const auto low           = static_cast<output_t>(input1);
  return ::cuda::std::array<output_t, 2>{high, low};
}

template <typename Input>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE auto merge_integers(const Input input_high, const Input input_low)
{
  static_assert(::cuda::std::is_integral_v<Input>);
  constexpr auto num_bits = ::cuda::std::__num_bits_v<Input>;
  using unsigned_t        = ::cuda::std::__make_nbit_uint_t<num_bits>;
  using unsigned_x2_t     = ::cuda::std::__make_nbit_uint_t<num_bits * 2>;
  using output_t          = ::cuda::std::__make_nbit_int_t<num_bits * 2, ::cuda::std::is_signed_v<Input>>;
  return static_cast<output_t>(
    (static_cast<unsigned_x2_t>(input_high) << num_bits) | static_cast<unsigned_t>(input_low));
}
} // namespace detail

CUB_NAMESPACE_END
