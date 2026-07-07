//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_COMPLEX_MATH_H
#define _CUDA_STD___SIMD_COMPLEX_MATH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__complex/complex.h>
#include <cuda/std/__complex/exponential_functions.h>
#include <cuda/std/__complex/hyperbolic_functions.h>
#include <cuda/std/__complex/inverse_hyperbolic_functions.h>
#include <cuda/std/__complex/inverse_trigonometric_functions.h>
#include <cuda/std/__complex/logarithms.h>
#include <cuda/std/__complex/math.h>
#include <cuda/std/__complex/roots.h>
#include <cuda/std/__complex/trigonometric_functions.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/type_traits.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.complex.math], helper functors for element-wise complex operations

struct __fn_real
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return __z.real();
  }
};

struct __fn_imag
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return __z.imag();
  }
};

struct __fn_abs
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::abs(__z);
  }
};

struct __fn_arg
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::arg(__z);
  }
};

struct __fn_norm
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::norm(__z);
  }
};

struct __fn_conj
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::conj(__z);
  }
};

struct __fn_proj
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::proj(__z);
  }
};

struct __fn_exp
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::exp(__z);
  }
};

struct __fn_log
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::log(__z);
  }
};

struct __fn_log10
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::log10(__z);
  }
};

struct __fn_sqrt
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::sqrt(__z);
  }
};

struct __fn_sin
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::sin(__z);
  }
};

struct __fn_asin
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::asin(__z);
  }
};

struct __fn_cos
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::cos(__z);
  }
};

struct __fn_acos
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::acos(__z);
  }
};

struct __fn_tan
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::tan(__z);
  }
};

struct __fn_atan
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::atan(__z);
  }
};

struct __fn_sinh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::sinh(__z);
  }
};

struct __fn_asinh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::asinh(__z);
  }
};

struct __fn_cosh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::cosh(__z);
  }
};

struct __fn_acosh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::acosh(__z);
  }
};

struct __fn_tanh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::tanh(__z);
  }
};

struct __fn_atanh
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __z) const noexcept
  {
    return ::cuda::std::atanh(__z);
  }
};

// Generic generator: applies a scalar functor to each element of a vec
template <typename _Vp, typename _Func>
struct __gen_complex_apply_unary
{
  const _Vp& __v_;
  _Func __func_ = {};

  template <typename _Idx>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(_Idx) const
  {
    return __func_(__v_[__simd_size_type{_Idx::value}]);
  }
};

// Generic binary generator: applies a scalar functor to corresponding elements of two vecs
template <typename _Vp, typename _Func>
struct __gen_complex_apply_binary
{
  const _Vp& __x_;
  const _Vp& __y_;
  _Func __func_ = {};

  template <typename _Idx>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(_Idx) const
  {
    return __func_(__x_[__simd_size_type{_Idx::value}], __y_[__simd_size_type{_Idx::value}]);
  }
};

// [simd.complex.math], unary complex functions returning real-valued result

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr rebind_t<__simd_complex_value_type_t<_Tp>, basic_vec<_Tp, _Abi>>
real(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<__simd_complex_value_type_t<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_unary<__vec_t, __fn_real>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr rebind_t<__simd_complex_value_type_t<_Tp>, basic_vec<_Tp, _Abi>>
imag(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<__simd_complex_value_type_t<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_unary<__vec_t, __fn_imag>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr rebind_t<__simd_complex_value_type_t<_Tp>, basic_vec<_Tp, _Abi>>
abs(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<__simd_complex_value_type_t<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_unary<__vec_t, __fn_abs>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr rebind_t<__simd_complex_value_type_t<_Tp>, basic_vec<_Tp, _Abi>>
arg(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<__simd_complex_value_type_t<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_unary<__vec_t, __fn_arg>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr rebind_t<__simd_complex_value_type_t<_Tp>, basic_vec<_Tp, _Abi>>
norm(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<__simd_complex_value_type_t<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_unary<__vec_t, __fn_norm>{__v}};
}

// [simd.complex.math], unary complex functions returning complex-valued result

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> conj(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_conj>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> proj(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_proj>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> exp(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_exp>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> log(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_log>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> log10(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_log10>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> sqrt(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_sqrt>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> sin(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_sin>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> asin(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_asin>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> cos(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_cos>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> acos(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_acos>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> tan(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_tan>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> atan(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_atan>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> sinh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_sinh>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> asinh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_asinh>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> cosh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_cosh>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> acosh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_acosh>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> tanh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_tanh>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi> atanh(const basic_vec<_Tp, _Abi>& __v)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_unary<__vec_t, __fn_atanh>{__v}};
}

// [simd.complex.math], binary complex function objects

struct __fn_polar_binary
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __rho, const _Tp& __theta) const
  {
    return ::cuda::std::polar(__rho, __theta);
  }
};

struct __fn_pow_binary
{
  template <typename _Tp>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(const _Tp& __x, const _Tp& __y) const
  {
    return ::cuda::std::pow(__x, __y);
  }
};

// [simd.complex.math], binary complex functions

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(::cuda::is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API rebind_t<::cuda::std::complex<_Tp>, basic_vec<_Tp, _Abi>>
polar(const basic_vec<_Tp, _Abi>& __x, const basic_vec<_Tp, _Abi>& __y = {})
{
  using __vec_t    = basic_vec<_Tp, _Abi>;
  using __result_t = rebind_t<::cuda::std::complex<_Tp>, __vec_t>;
  return __result_t{__gen_complex_apply_binary<__vec_t, __fn_polar_binary>{__x, __y}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi)
_CCCL_REQUIRES(__is_complex_vectorizable_v<_Tp>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr basic_vec<_Tp, _Abi>
pow(const basic_vec<_Tp, _Abi>& __x, const basic_vec<_Tp, _Abi>& __y)
{
  using __vec_t = basic_vec<_Tp, _Abi>;
  return __vec_t{__gen_complex_apply_binary<__vec_t, __fn_pow_binary>{__x, __y}};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_COMPLEX_MATH_H
