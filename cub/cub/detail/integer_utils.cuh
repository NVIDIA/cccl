/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/
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

#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cmath> // isnan
#include <cuda/std/limits> // numeric_limits
#include <cuda/std/type_traits> // __make_nbit_int_t
#include <cuda/type_traits> // is_floating_point_v

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * Integer Utils
 **********************************************************************************************************************/

namespace detail
{

template <typename Input>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto split_integers(Input input)
{
  using namespace _CUDA_VSTD;
  static_assert(is_integral_v<Input>);
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
  using namespace _CUDA_VSTD;
  static_assert(is_integral_v<Input>);
  constexpr auto num_bits = __num_bits_v<Input>;
  using unsigned_t        = __make_nbit_uint_t<num_bits>;
  using unsigned_X2_t     = __make_nbit_uint_t<num_bits * 2>;
  using output_t          = __make_nbit_int_t<num_bits * 2, is_signed_v<Input>>;
  return static_cast<output_t>((static_cast<unsigned_X2_t>(inputA) << num_bits) | static_cast<unsigned_t>(inputB));
}

template <typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE auto comparable_int_to_floating_point(T value)
{
  static_assert(_CUDA_VSTD::is_integral_v<T>);
  constexpr auto lowest = T{1} << (_CUDA_VSTD::__num_bits_v<T> - 1);
  return static_cast<T>(value < 0 ? lowest - value : value);
}

template <typename ReductionOp, typename T>
_CCCL_DEVICE _CCCL_FORCEINLINE auto floating_point_to_comparable_int(ReductionOp, T value)
{
  using namespace _CUDA_VSTD;
  static_assert(::cuda::is_floating_point_v<T>);
  static_assert(is_cuda_minimum_maximum_v<ReductionOp, T>);
  using signed_t        = __make_nbit_int_t<__num_bits_v<T>, true>;
  constexpr auto lowest = signed_t{1} << (__num_bits_v<T> - 1);
  constexpr auto is_max = is_cuda_maximum_v<ReductionOp, T>;
  const auto nan        = is_max ? -numeric_limits<T>::quiet_NaN() : numeric_limits<T>::quiet_NaN();
  auto value1           = _CUDA_VSTD::isnan(value) ? nan : value;
  auto value_int        = cub::detail::unsafe_bitcast<signed_t>(value1);
  return static_cast<signed_t>(value_int < 0 ? lowest - value_int : value_int);
}

} // namespace detail
CUB_NAMESPACE_END
