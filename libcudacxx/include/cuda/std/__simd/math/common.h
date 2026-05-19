//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_MATH_COMMON_H
#define _CUDA_STD___SIMD_MATH_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/type_traits.h> // rebind_t
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// concept simd-vec-type, exposition only
template <typename _Tp, typename _Up = remove_cvref_t<_Tp>>
_CCCL_CONCEPT __is_simd_vec_type_v = _CCCL_REQUIRES_EXPR((_Up))(
  typename(typename _Up::value_type),
  typename(typename _Up::abi_type),
  requires(is_same_v<_Up, basic_vec<typename _Up::value_type, typename _Up::abi_type>>),
  requires(is_default_constructible_v<_Up>));

template <typename _Tp, typename = void>
struct __deduced_vec
{};

template <typename _Tp>
struct __deduced_vec<_Tp, void_t<decltype(::cuda::std::declval<const _Tp&>() + ::cuda::std::declval<const _Tp&>())>>
{
  using type = decltype(::cuda::std::declval<const _Tp&>() + ::cuda::std::declval<const _Tp&>());
};

// using deduced-vec-t, exposition only
template <typename _Tp>
using __deduced_vec_t = typename __deduced_vec<_Tp>::type;

// concept simd-floating-point, exposition only
template <typename _Tp>
_CCCL_CONCEPT __is_simd_floating_point_v = _CCCL_REQUIRES_EXPR(
  (_Tp))(typename(typename _Tp::value_type),
         requires(__is_simd_vec_type_v<_Tp>),
         requires(::cuda::is_floating_point_v<typename _Tp::value_type>));

// concept math-floating-point, exposition only
template <typename _Tp>
_CCCL_CONCEPT __is_math_floating_point_v = _CCCL_REQUIRES_EXPR(
  (_Tp))(typename(__deduced_vec_t<_Tp>), requires(__is_simd_floating_point_v<__deduced_vec_t<_Tp>>));

//----------------------------------------------------------------------------------------------------------------------
// unary macros

// common macro to implement unary generator
#define _CCCL_SIMD_MATH_UNARY_GENERATOR(_NAME)                                  \
  template <typename _Result, typename _Vp>                                     \
  struct __simd_##_NAME##_generator                                             \
  {                                                                             \
    using __result_t = typename _Result::value_type;                            \
                                                                                \
    const _Vp& __x_;                                                            \
                                                                                \
    template <typename _Ip>                                                     \
    [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept \
    {                                                                           \
      return static_cast<__result_t>(::cuda::std::_NAME(__x_[_Ip::value]));     \
    }                                                                           \
  }

// common macro to implement unary function
#define _CCCL_SIMD_MATH_UNARY_FUNCTION(_NAME, _CONSTEXPR)                   \
  _CCCL_SIMD_MATH_UNARY_GENERATOR(_NAME);                                   \
                                                                            \
  _CCCL_TEMPLATE(typename _Vp, typename _Result = __deduced_vec_t<_Vp>)     \
  _CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)                           \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp& __x) noexcept \
  {                                                                         \
    return _Result{__simd_##_NAME##_generator<_Result, _Vp>{__x}};          \
  }

// common macro to implement (unary) rebind function
#define _CCCL_SIMD_MATH_UNARY_REBIND_FUNCTION(_NAME, _Tp, _CONSTEXPR)                  \
  _CCCL_TEMPLATE(typename _Vp, typename _Result = rebind_t<_Tp, __deduced_vec_t<_Vp>>) \
  _CCCL_REQUIRES(__is_math_floating_point_v<_Vp>)                                      \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp& __x) noexcept            \
  {                                                                                    \
    return _Result{__simd_##_NAME##_generator<_Result, _Vp>{__x}};                     \
  }

//----------------------------------------------------------------------------------------------------------------------

// The following overloads don't work with nvcc (while they work with clang and gcc) because they are recognized as
// ambiguous with the above overload. The workaround consists in checking if the non-floating point type is convertible
// to the floating point type.

// For example:
//  template<math-floating-point V>
//  constexpr deduced-vec-t<V> pow(const V& x, const V& y);

//  template<math-floating-point V>
//  constexpr deduced-vec-t<V> pow(const deduced-vec-t<V>& x, const V& y);

//  template<math-floating-point V>
//  constexpr deduced-vec-t<V> pow(const V& x, const deduced-vec-t<V>& y);

template <typename _Void, typename... _Args>
struct __simd_math_result
{};

template <typename... _Args>
struct __simd_math_result<void_t<decltype((::cuda::std::declval<const _Args&>() + ...))>, _Args...>
{
  using type = decltype((::cuda::std::declval<const _Args&>() + ...));
};

template <typename... _Args>
using __simd_math_result_t = typename __simd_math_result<void, _Args...>::type;

template <typename _Arg, typename _Result>
inline constexpr bool __is_simd_math_same_vec_arg_v = is_same_v<__deduced_vec_t<_Arg>, _Result>;

template <typename _Arg, typename _Result>
inline constexpr bool __is_simd_math_arg_v =
  __is_simd_math_same_vec_arg_v<_Arg, _Result>
  || (!__is_math_floating_point_v<_Arg> && is_convertible_v<const _Arg&, _Result>);

template <typename _Result, typename... _Args>
inline constexpr bool __is_simd_math_v =
  __is_simd_floating_point_v<_Result> && (__is_simd_math_arg_v<_Args, _Result> && ...);

//----------------------------------------------------------------------------------------------------------------------
// binary macros

// common macro to implement a binary generator
#define _CCCL_SIMD_MATH_BINARY_GENERATOR(_NAME, _GENERATOR)                                   \
  template <typename _Result, typename _Vp0, typename _Vp1>                                   \
  struct __simd_##_GENERATOR##_generator                                                      \
  {                                                                                           \
    using __result_t = typename _Result::value_type;                                          \
                                                                                              \
    const _Vp0& __x_;                                                                         \
    const _Vp1& __y_;                                                                         \
                                                                                              \
    template <typename _Ip>                                                                   \
    [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept               \
    {                                                                                         \
      return static_cast<__result_t>(::cuda::std::_NAME(__x_[_Ip::value], __y_[_Ip::value])); \
    }                                                                                         \
  }

#define _CCCL_SIMD_MATH_BINARY_FUNCTION(_NAME, _GENERATOR, _CONSTEXPR)                              \
  _CCCL_TEMPLATE(typename _Vp0, typename _Vp1, typename _Result = __simd_math_result_t<_Vp0, _Vp1>) \
  _CCCL_REQUIRES(__is_simd_math_v<_Result, _Vp0, _Vp1>)                                             \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp0& __x, const _Vp1& __y) noexcept       \
  {                                                                                                 \
    const _Result __x_vec = __x;                                                                    \
    const _Result __y_vec = __y;                                                                    \
    return _Result{__simd_##_GENERATOR##_generator<_Result, _Result, _Result>{__x_vec, __y_vec}};   \
  }

//----------------------------------------------------------------------------------------------------------------------
// ternary macros

#define _CCCL_SIMD_MATH_TERNARY_GENERATOR(_NAME, _GENERATOR)                                                    \
  template <typename _Result, typename _Vp0, typename _Vp1, typename _Vp2>                                      \
  struct __simd_##_GENERATOR##_generator                                                                        \
  {                                                                                                             \
    using __result_t = typename _Result::value_type;                                                            \
                                                                                                                \
    const _Vp0& __x_;                                                                                           \
    const _Vp1& __y_;                                                                                           \
    const _Vp2& __z_;                                                                                           \
                                                                                                                \
    template <typename _Ip>                                                                                     \
    [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Ip) const noexcept                                 \
    {                                                                                                           \
      return static_cast<__result_t>(::cuda::std::_NAME(__x_[_Ip::value], __y_[_Ip::value], __z_[_Ip::value])); \
    }                                                                                                           \
  }

#define _CCCL_SIMD_MATH_TERNARY_FUNCTION(_NAME, _GENERATOR, _CONSTEXPR)                                             \
  _CCCL_TEMPLATE(                                                                                                   \
    typename _Vp0, typename _Vp1, typename _Vp2, typename _Result = __simd_math_result_t<_Vp0, _Vp1, _Vp2>)         \
  _CCCL_REQUIRES(__is_simd_math_v<_Result, _Vp0, _Vp1, _Vp2>)                                                       \
  [[nodiscard]] _CCCL_API _CONSTEXPR _Result _NAME(const _Vp0& __x, const _Vp1& __y, const _Vp2& __z) noexcept      \
  {                                                                                                                 \
    const _Result __x_vec = __x;                                                                                    \
    const _Result __y_vec = __y;                                                                                    \
    const _Result __z_vec = __z;                                                                                    \
    return _Result{__simd_##_GENERATOR##_generator<_Result, _Result, _Result, _Result>{__x_vec, __y_vec, __z_vec}}; \
  }

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_MATH_COMMON_H
