/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * Integer Utils
 **********************************************************************************************************************/

namespace detail
{

template <typename Input>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto split_integers(Input input)
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
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE auto merge_integers(Input inputA, Input inputB)
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
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE auto make_floating_point_int_comparible(T value)
{
  static_assert(_CUDA_VSTD::is_integral_v<T>);
  constexpr auto lowest = T{1} << (_CUDA_VSTD::__num_bits_v<T> - 1);
  return static_cast<T>(value < 0 ? lowest - value : value);
}

} // namespace detail

CUB_NAMESPACE_END
