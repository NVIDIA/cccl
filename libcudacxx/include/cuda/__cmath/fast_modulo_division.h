//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
#define _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/num_bits.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

/***********************************************************************************************************************
 * Extract higher bits after multiplication
 **********************************************************************************************************************/

template <typename _Tp, typename _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _CUDA_VSTD::common_type_t<_Tp, _Up>
__multiply_extract_higher_bits(_Tp __x, _Up __y)
{
  using namespace _CUDA_VSTD;
  static_assert(__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= 8, "unsupported type");
  static_assert(__cccl_is_integer_v<_Up> && sizeof(_Up) <= 8, "unsupported type");
  if constexpr (is_signed_v<_Tp>)
  {
    _CCCL_ASSERT(__x >= 0, "__x must be non-negative");
  }
  if constexpr (is_signed_v<_Up>)
  {
    _CCCL_ASSERT(__y >= 0, "__y must be non-negative");
  }
  constexpr auto __mul_bits = ::cuda::next_power_of_two(__num_bits_v<_Tp> + __num_bits_v<_Up>);
  using __larger_t          = __make_nbit_uint_t<__mul_bits>;
  using __ret_t             = common_type_t<_Tp, _Up>;
  auto __ret                = (static_cast<__larger_t>(__x) * __y) >> (__mul_bits / 2);
  return static_cast<__ret_t>(__ret);
}

/***********************************************************************************************************************
 * Fast Modulo/Division based on Precomputation
 **********************************************************************************************************************/

template <typename _Tp>
struct div_t
{
  _Tp quotient;
  _Tp remainder;
};

template <typename _Tp>
class fast_mod_div
{
  static_assert(_CUDA_VSTD::__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= 8, "unsupported type");

  using __unsigned_t = _CUDA_VSTD::make_unsigned_t<_Tp>;

public:
  fast_mod_div() = delete;

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI explicit fast_mod_div(_Tp __divisor1) noexcept
      : __divisor{__divisor1}
      , __shift_right{::cuda::ceil_ilog2(__divisor)}
  {
    using namespace _CUDA_VSTD;
    using __larger_t = __make_nbit_uint_t<__num_bits_v<_Tp> * 2>;
    auto __exp       = __num_bits_v<_Tp> + __shift_right;
    __multiplier     = static_cast<_Tp>(::cuda::ceil_div(__larger_t{1} << __exp, __divisor));
    //_CCCL_ASSERT(__shift_right < __num_bits_v<__common_t>, "wrong __shift_right");
  }

  fast_mod_div(const fast_mod_div&) noexcept = default;

  fast_mod_div(fast_mod_div&&) noexcept = default;

  template <typename _Up>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE div_t<_CUDA_VSTD::common_type_t<_Tp, _Up>>
  __divide(_Up __dividend) const noexcept
  {
    using namespace _CUDA_VSTD;
    static_assert(__cccl_is_integer_v<_Tp> && sizeof(_Tp) <= 8, "unsupported type");
    using __common_t = common_type_t<_Tp, _Up>;
    using __result_t = div_t<__common_t>;
    if constexpr (is_signed_v<_Up>)
    {
      _CCCL_ASSERT(__dividend >= 0, "dividend must be non-negative");
    }
    // using __unsigned_Tp_t = make_unsigned_t<_Tp>;
    // using __unsigned_Up_t = make_unsigned_t<_Up>;
    // auto __divisor1       = static_cast<__unsigned_Tp_t>(__divisor);
    // auto __multiplier1    = static_cast<__unsigned_Tp_t>(__multiplier);
    // auto __dividend1      = static_cast<__unsigned_Up_t>(__dividend);
    //  we don't need to check if divided >= 2^N-1 if it is signed or it is a smaller type
    if (is_unsigned_v<_Tp> && sizeof(_Up) >= sizeof(_Tp) && __dividend >= numeric_limits<_Tp>::max() / 2)
    {
      auto __quotient = (__dividend >= __divisor);
      auto __reminder = __quotient ? __dividend - __divisor : __dividend; // or __dividend - (__quotient * __divisor)
      return __result_t{static_cast<__common_t>(__quotient), static_cast<__common_t>(__reminder)};
    }
    // auto __higher_bits =
    //   (__multiplier == 0) ? __dividend : ::cuda::__multiply_extract_higher_bits(__dividend, __multiplier);
    auto __higher_bits = ::cuda::__multiply_extract_higher_bits(__dividend, __multiplier);
    auto __quotient    = __higher_bits >> __shift_right;
    auto __remainder   = __dividend - (__quotient * __divisor);
    _CCCL_ASSERT(__quotient == __dividend / __divisor, "wrong __quotient");
    _CCCL_ASSERT(__remainder == __dividend % __divisor, "wrong __remainder");
    return __result_t{static_cast<__common_t>(__quotient), static_cast<__common_t>(__remainder)};
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI operator _Tp() const noexcept
  {
    return static_cast<_Tp>(__divisor);
  }

private:
  _Tp __divisor          = 1;
  _Tp __multiplier       = 0;
  uint32_t __shift_right = 0;
};

/***********************************************************************************************************************
 * Non-member functions
 **********************************************************************************************************************/

template <typename _Tp, typename _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI div_t<_CUDA_VSTD::common_type_t<_Tp, _Up>>
div(_Tp __dividend, fast_mod_div<_Up> __divisor) noexcept
{
  return __divisor.__divide(__dividend);
}

template <typename _Tp, typename _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::common_type_t<_Tp, _Up>
operator/(_Tp __dividend, fast_mod_div<_Up> __divisor) noexcept
{
  return __divisor.__divide(__dividend).__quotient;
}

template <typename _Tp, typename _Up>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI _CUDA_VSTD::common_type_t<_Tp, _Up>
operator%(_Tp __dividend, fast_mod_div<_Up> __divisor) noexcept
{
  return __divisor.__divide(__dividend).__remainder;
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CMATH_FAST_MODULO_DIVISION_H
