//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MDSPAN_FOR_EACH_IN_EXTENT_H
#define _CUDA__MDSPAN_FOR_EACH_IN_EXTENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned
__multiply_extract_higher_bits(unsigned __dividend, unsigned __multiplier) noexcept
{
  // this optimization is obsolete for recent architectures/compilers
  // clang-format off
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    (return __umulhi(__dividend, __multiplier);),
    (return static_cast<unsigned>((static_cast<::cuda::std::uint64_t>(__dividend) * __multiplier) >> 32u);)
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
      : __divisor{divisor}
  {
    _CCCL_ASSERT(divisor > 0, "divisor must be positive");
    if (divisor == 1)
    {
      return;
    }
    auto num_bits = ::cuda::std::bit_width(divisor) + !::cuda::std::has_single_bit(divisor);
    __multiplier  = static_cast<unsigned>(::cuda::ceil_div(int64_t{1} << (num_bits + 30), divisor));
    __shift_right = num_bits - 2;
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result operator()(unsigned dividend) const noexcept
  {
    if (__divisor == 1)
    {
      return result{dividend, 0};
    }
    auto quotient  = __multiply_extract_higher_bits(dividend, __multiplier) >> __shift_right;
    auto remainder = dividend - (quotient * __divisor);
    assert(remainder >= 0 && remainder < __divisor);
    return result{quotient, remainder};
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend unsigned
  operator/(unsigned dividend, fast_div_mod __div) noexcept
  {
    return __div(dividend).quotient;
  }

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE friend unsigned
  operator%(unsigned dividend, fast_div_mod __div) noexcept
  {
    return __div(dividend).remainder;
  }

private:
  unsigned __divisor     = 1;
  unsigned __multiplier  = 0;
  unsigned __shift_right = 0;
};

/***********************************************************************************************************************
 * Utilities
 **********************************************************************************************************************/

template <::cuda::std::size_t _Rank, typename _IndexType, ::cuda::std::size_t... _Extents, ::cuda::std::size_t... _Indices>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::size_t
sub_size(const ::cuda::std::extents<_IndexType, _Extents...>& __ext, ::cuda::std::index_sequence<_Indices...> = {})
{
  if constexpr (_Rank >= __ext.rank())
  {
    return _IndexType{1};
  }
  else if constexpr (sizeof...(_Indices) == 0)
  {
    return sub_size<_Rank>(__ext, ::cuda::std::make_index_sequence<sizeof...(_Extents) - _Rank>{});
  }
  else
  {
    return (__ext.extent(_Rank + _Indices) * ...);
  }
}

template <typename _IndexType, ::cuda::std::size_t... _E, ::cuda::std::size_t... _Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto sub_sizes_fast_div_mod(
  const ::cuda::std::extents<_IndexType, _E...>& __ext, ::cuda::std::index_sequence<_Ranks...> = {})
{
  return ::cuda::std::array{fast_div_mod{sub_size<_Ranks + 1>(__ext)}...};
}

template <typename _IndexType, ::cuda::std::size_t... _E, ::cuda::std::size_t... _Ranks>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto
extends_fast_div_mod(const ::cuda::std::extents<_IndexType, _E...>& __ext, ::cuda::std::index_sequence<_Ranks...> = {})
{
  return ::cuda::std::array{fast_div_mod{__ext.extent(_Ranks)}...};
}

/***********************************************************************************************************************
 * for_each_in_extent
 **********************************************************************************************************************/

template <int Rank, typename T>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE//
T coordinate_at(T index, fast_div_mod sub_size, fast_div_mod extent)
{
  return (index / sub_size) % extent;
}

// Due to the limitations of CUDA kernel, we cannot use more than one parameter pack
template <typename Func, typename FastDivModArrayType, ::cuda::std::size_t... Ranks>
__global__ void for_each_in_extent_kernel(
  __grid_constant__ const Func func,
  __grid_constant__ const FastDivModArrayType sub_sizes_div_array,
  __grid_constant__ const FastDivModArrayType extends_div_array,
  ::cuda::std::index_sequence<Ranks...> = {})
{
  auto id = threadIdx.x + blockIdx.x * blockDim.x;
  func(coordinate_at<Ranks>(id, sub_sizes_div_array[Ranks], extends_div_array[Ranks])...);
}

template <class _IndexType, ::cuda::std::size_t... _Extents, typename Func>
void for_each_in_extent(const ::cuda::std::extents<_IndexType, _Extents...>& ext, const Func& func)
{
  constexpr auto seq                     = ::cuda::std::make_index_sequence<sizeof...(_Extents)>{};
  ::cuda::std::array sub_sizes_div_array = sub_sizes_fast_div_mod(ext, seq);
  ::cuda::std::array extends_div_array   = extends_fast_div_mod(ext, seq);
  for_each_in_extent_kernel<<<1, sub_size<0>(ext)>>>(ext, func, sub_sizes_div_array, extends_div_array, seq);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA__MDSPAN_FOR_EACH_IN_EXTENT_H
