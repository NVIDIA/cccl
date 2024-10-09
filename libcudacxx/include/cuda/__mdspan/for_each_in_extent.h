//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CMATH_FAST_DIV_MOD
#define _CUDA___CMATH_FAST_DIV_MOD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr unsigned
__multiply_extract_higher_bits(unsigned __dividend, unsigned __multiplier) noexcept
{
  // clang-format off
  NV_IF_ELSE_TARGET(NV_IS_DEVICE,
    (return __umulhi(__dividend, __multiplier);),
    (return static_cast<unsigned>((static_cast<_CUDA_VSTD::uint64_t>(__dividend) * __multiplier) >> 32u);)
  )
  // clang-format on
}

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
    if (divisor == 1)
    {
      return;
    }
    auto num_bits = _CUDA_VSTD::bit_width(divisor);
    __multiplier  = static_cast<unsigned>(::cuda::ceil_div(int64_t{1} << (num_bits + 31), divisor));
    __shift_right = num_bits - 1;
  }

  fast_div_mod(const fast_div_mod&) noexcept = default;

  fast_div_mod(fast_div_mod&&) noexcept = default;

  _CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE result operator()(unsigned dividend) const noexcept
  {
    auto quotient =
      (__divisor != 1) ? __multiply_extract_higher_bits(dividend, __multiplier) >> __shift_right : dividend;
    auto remainder = dividend - (quotient * __divisor);
    return result{quotient, remainder};
  }

private:
  unsigned __divisor     = 1;
  unsigned __multiplier  = 0;
  unsigned __shift_right = 0;
};

/***********************************************************************************************************************
 * sub_size
 **********************************************************************************************************************/

template <_CUDA_VSTD::size_t _Rank, typename _IndexType, _CUDA_VSTD::size_t... _Extents, _CUDA_VSTD::size_t... _Indices>
_CCCL_NODISCARD _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr _CUDA_VSTD::size_t
sub_size(const std::extents<_IndexType, _Extents...>& __ext, _CUDA_VSTD::index_sequence<_Indices...> = {})
{
  if constexpr (sizeof...(_Extents) == 0)
  {
    return sub_size<_Rank>(__ext, _CUDA_VSTD::make_index_sequence<sizeof...(_Extents) - _Rank>{});
  }
  else
  {
    return (__ext.extent(_Rank + _Indices) * ...);
  }
}

/***********************************************************************************************************************
 * for_each_in_extent
 **********************************************************************************************************************/

template <typename _IndexType,
          std::size_t... _Extents,
          typename Func,
          std::size_t FirstExtInd,
          std::size_t... RestExtInds,
          typename... Is>
void __for_each_in_extent_impl(
  _IndexType thread_id,
  fast_div_mod (&fast_div_mod_array)[sizeof...(_Extents)],
  const std::extents<_IndexType, _Extents...>& ext,
  Func&& func,
  std::index_sequence<FirstExtInd, RestExtInds...>,
  Is... is)
{
  if constexpr (sizeof...(RestExtInds) > 0)
  {
    constexpr auto __rank                    = sizeof...(RestExtInds);
    auto [thread_group_id, thread_group_rem] = fast_div_mod_array[__rank](thread_id);
    for_each_in_extent(ext, std::forward<Func>(func), std::index_sequence<RestExtInds...>{}, is..., thread_group_rem);
  }
  else
  {
    std::forward<Func>(func)(is..., thread_id);
  }
}

template <class _IndexType, std::size_t... _Extents, typename Func>
void for_each_in_extent(stdex::extents<_IndexType, _Extents...> ext, Func&& func)
{
  for_each_in_extent(ext, std::forward<Func>(func), std::make_index_sequence<sizeof...(_Extents)>{});
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___CMATH_FAST_DIV_MOD
