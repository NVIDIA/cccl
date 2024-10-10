/******************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h> // cuda::std::ceil_div
#include <cuda/std/__bit/has_single_bit.h> // std::has_single_bit
#include <cuda/std/__bit/integral.h> // std::bit_width
#include <cuda/std/cstdint> // uint64_t

CUB_NAMESPACE_BEGIN

namespace detail
{

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned
multiply_extract_higher_bits(unsigned dividend, unsigned _multiplier) noexcept
{
  // this optimization is obsolete for recent architectures/compilers
  // clang-format off
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    (return __umulhi(dividend, _multiplier);),
    (return static_cast<unsigned>((static_cast<::cuda::std::uint64_t>(dividend) * _multiplier) >> 32u);)
  )
  // clang-format on
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

struct fast_div_mod
{
  struct result
  {
    unsigned quotient;
    unsigned remainder;
  };

  fast_div_mod() = delete;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE explicit fast_div_mod(unsigned divisor) noexcept
      : _divisor{divisor}
  {
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    if (divisor == 1)
    {
      return;
    }
    auto num_bits = ::cuda::std::bit_width(divisor) + !::cuda::std::has_single_bit(divisor);
    _multiplier   = static_cast<unsigned>(::cuda::ceil_div(int64_t{1} << (num_bits + 30), divisor));
    _shift_right  = num_bits - 2;
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result operator()(unsigned dividend) const noexcept
  {
    if (_divisor == 1)
    {
      return result{dividend, 0};
    }
    auto quotient  = multiply_extract_higher_bits(dividend, _multiplier) >> _shift_right;
    auto remainder = dividend - (quotient * _divisor);
    _CCCL_ASSERT(remainder >= 0 && remainder < _divisor, "remainder out of range");
    return result{quotient, remainder};
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend unsigned
  operator/(unsigned dividend, fast_div_mod div) noexcept
  {
    return div(dividend).quotient;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend unsigned
  operator%(unsigned dividend, fast_div_mod div) noexcept
  {
    return div(dividend).remainder;
  }

private:
  unsigned _divisor     = 1;
  unsigned _multiplier  = 0;
  unsigned _shift_right = 0;
};

} // namespace detail

CUB_NAMESPACE_END
