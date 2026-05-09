//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ARGUMENT_ARGUMENT_H
#define _CUDA___ARGUMENT_ARGUMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__argument/argument_bounds.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__fwd/span.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_ARGUMENT

// =====================================================================
// __element_type_of — extract element type from arg type
// =====================================================================

template <class _Tp>
struct __element_type_of
{
  using type = _Tp;
};

template <class _Tp, size_t _Extent>
struct __element_type_of<::cuda::std::span<_Tp, _Extent>>
{
  using type = _Tp;
};

template <class _Tp, size_t _Size>
struct __element_type_of<::cuda::std::array<_Tp, _Size>>
{
  using type = _Tp;
};

template <class _Tp>
using __element_type_of_t = typename __element_type_of<_Tp>::type;

// =====================================================================
// constant
// =====================================================================

//! @brief Wraps a compile-time constant argument value.
template <auto _Value>
struct constant
{
  using value_type                  = decltype(_Value);
  using __element_type              = __element_type_of_t<value_type>;
  static constexpr value_type value = _Value;
};

// =====================================================================
// dynamic
// =====================================================================

//! @brief Wraps a runtime argument value with optional bounds.
template <class _Arg, class _StaticBounds = __no_bounds>
struct dynamic
{
  using __element_type = __element_type_of_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  runtime_bounds<__element_type> __runtime_bounds_;

private:
  _CCCL_API constexpr void __validate_element(__element_type __val) noexcept
  {
    if constexpr (!::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
    {
      _CCCL_ASSERT(__val >= __static_bounds_.lowest, "dynamic argument value is below static lowest bound");
      _CCCL_ASSERT(__val <= __static_bounds_.max, "dynamic argument value is above static max bound");
    }
    _CCCL_ASSERT(__val >= __runtime_bounds_.lowest, "dynamic argument value is below runtime lowest bound");
    _CCCL_ASSERT(__val <= __runtime_bounds_.max, "dynamic argument value is above runtime max bound");
  }

  _CCCL_API constexpr void __validate() noexcept
  {
    if constexpr (::cuda::std::is_arithmetic_v<_Arg> || ::cuda::std::is_enum_v<_Arg>)
    {
      __validate_element(arg);
    }
    else if constexpr (::cuda::std::__is_cuda_std_span_v<_Arg>)
    {
      for (size_t __i = 0; __i < arg.size(); ++__i)
      {
        __validate_element(arg[__i]);
      }
    }
  }

public:
  _CCCL_API constexpr dynamic(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Lowest, auto _Max>
  _CCCL_API constexpr dynamic(_Arg __arg, static_bounds<_Lowest, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {
    __validate();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr dynamic(_Arg __arg, runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {
    static_assert(!::cuda::std::is_same_v<_Arg, __element_type>,
                  "runtime bounds on a single-value dynamic argument are not supported; use static bounds instead");
    __validate();
  }

  template <auto _Lowest, auto _Max, class _BoundsTp>
  _CCCL_API constexpr dynamic(_Arg __arg, static_bounds<_Lowest, _Max> __sb, runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {
    static_assert(!::cuda::std::is_same_v<_Arg, __element_type>,
                  "runtime bounds on a single-value dynamic argument are not supported; use static bounds instead");
    __validate();
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE dynamic(_Arg) -> dynamic<_Arg>;

template <class _Arg, auto _Lowest, auto _Max>
_CCCL_HOST_DEVICE dynamic(_Arg, static_bounds<_Lowest, _Max>) -> dynamic<_Arg, static_bounds<_Lowest, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE dynamic(_Arg, runtime_bounds<_Tp>) -> dynamic<_Arg>;

template <class _Arg, auto _Lowest, auto _Max, class _Tp>
_CCCL_HOST_DEVICE dynamic(_Arg, static_bounds<_Lowest, _Max>, runtime_bounds<_Tp>)
  -> dynamic<_Arg, static_bounds<_Lowest, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// deferred
// =====================================================================

//! @brief Wraps a device-resident argument value that is not host-accessible at API call time.
template <class _Arg, class _StaticBounds = __no_bounds>
struct deferred
{
  static_assert(::cuda::std::__is_cuda_std_span_v<_Arg>, "deferred argument requires a cuda::std::span");

  using __element_type = __element_type_of_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  runtime_bounds<__element_type> __runtime_bounds_;

  _CCCL_API constexpr deferred(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Lowest, auto _Max>
  _CCCL_API constexpr deferred(_Arg __arg, static_bounds<_Lowest, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {}

  template <class _BoundsTp>
  _CCCL_API constexpr deferred(_Arg __arg, runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {}

  template <auto _Lowest, auto _Max, class _BoundsTp>
  _CCCL_API constexpr deferred(_Arg __arg, static_bounds<_Lowest, _Max> __sb, runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {}
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE deferred(_Arg) -> deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Max>
_CCCL_HOST_DEVICE deferred(_Arg, static_bounds<_Lowest, _Max>) -> deferred<_Arg, static_bounds<_Lowest, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE deferred(_Arg, runtime_bounds<_Tp>) -> deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Max, class _Tp>
_CCCL_HOST_DEVICE deferred(_Arg, static_bounds<_Lowest, _Max>, runtime_bounds<_Tp>)
  -> deferred<_Arg, static_bounds<_Lowest, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __is_single_value_v
// =====================================================================

template <class _Tp>
inline constexpr bool __is_value_ref_v = false;
template <class _Tp>
inline constexpr bool __is_value_ref_v<::cuda::std::span<_Tp, 1>> = true;

template <class _Tp>
inline constexpr bool __is_single_value_v =
  ::cuda::std::is_arithmetic_v<::cuda::std::remove_cv_t<_Tp>> || ::cuda::std::is_enum_v<::cuda::std::remove_cv_t<_Tp>>
  || __is_value_ref_v<_Tp>;

// =====================================================================
// unwrap
// =====================================================================

template <class _Tp>
inline constexpr bool __is_wrapper_v = false;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<dynamic<_Arg, _StaticBounds>> = true;
template <auto _Value>
inline constexpr bool __is_wrapper_v<constant<_Value>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<deferred<_Arg, _StaticBounds>> = true;

template <
  class _Tp,
  ::cuda::std::enable_if_t<!__is_wrapper_v<::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<_Tp>>>, int> = 0>
[[nodiscard]] _CCCL_API constexpr _Tp&& unwrap(_Tp&& __arg) noexcept
{
  return ::cuda::std::forward<_Tp>(__arg);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& unwrap(const dynamic<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr const decltype(_Value)& unwrap(const constant<_Value>&) noexcept
{
  return constant<_Value>::value;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& unwrap(const deferred<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

// =====================================================================
// Internal helpers for bounds computation
// =====================================================================

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_lowest() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::lowest();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::lowest);
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_max() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::max();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::max);
  }
}

template <auto _Value>
inline constexpr auto __constant_compute_lowest = [] {
  using _VT = decltype(_Value);
  using _ET = __element_type_of_t<_VT>;
  if constexpr (::cuda::std::is_same_v<_VT, _ET>)
  {
    return _Value;
  }
  else
  {
    auto __result = _Value[0];
    for (size_t __i = 1; __i < _Value.size(); ++__i)
    {
      if (_Value[__i] < __result)
      {
        __result = _Value[__i];
      }
    }
    return __result;
  }
}();

template <auto _Value>
inline constexpr auto __constant_compute_max = [] {
  using _VT = decltype(_Value);
  using _ET = __element_type_of_t<_VT>;
  if constexpr (::cuda::std::is_same_v<_VT, _ET>)
  {
    return _Value;
  }
  else
  {
    auto __result = _Value[0];
    for (size_t __i = 1; __i < _Value.size(); ++__i)
    {
      if (_Value[__i] > __result)
      {
        __result = _Value[__i];
      }
    }
    return __result;
  }
}();

// =====================================================================
// traits
// =====================================================================

//! @brief Traits for argument wrappers and plain argument values.
template <class _Tp>
struct traits
{
  using value_type                       = _Tp;
  using __element_type                   = __element_type_of_t<_Tp>;
  static constexpr bool is_deferred      = false;
  static constexpr __element_type lowest = ::cuda::std::numeric_limits<__element_type>::lowest();
  static constexpr __element_type max    = ::cuda::std::numeric_limits<__element_type>::max();
};

template <class _Arg, class _StaticBounds>
struct traits<dynamic<_Arg, _StaticBounds>>
{
  using value_type                       = _Arg;
  using __element_type                   = __element_type_of_t<_Arg>;
  static constexpr bool is_deferred      = false;
  static constexpr __element_type lowest = __wrapper_static_lowest<__element_type, _StaticBounds>();
  static constexpr __element_type max    = __wrapper_static_max<__element_type, _StaticBounds>();
};

template <auto _Value>
struct traits<constant<_Value>>
{
  using value_type                       = decltype(_Value);
  using __element_type                   = __element_type_of_t<value_type>;
  static constexpr bool is_deferred      = false;
  static constexpr __element_type lowest = __constant_compute_lowest<_Value>;
  static constexpr __element_type max    = __constant_compute_max<_Value>;
};

template <class _Arg, class _StaticBounds>
struct traits<deferred<_Arg, _StaticBounds>>
{
  using value_type                       = _Arg;
  using __element_type                   = __element_type_of_t<_Arg>;
  static constexpr bool is_deferred      = true;
  static constexpr __element_type lowest = __wrapper_static_lowest<__element_type, _StaticBounds>();
  static constexpr __element_type max    = __wrapper_static_max<__element_type, _StaticBounds>();
};

// =====================================================================
// lowest / max — free functions (effective bounds combining static + runtime)
// =====================================================================

template <class _Tp, ::cuda::std::enable_if_t<!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>, int> = 0>
[[nodiscard]] _CCCL_API constexpr auto lowest(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::lowest();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto lowest(constant<_Value>) noexcept
{
  return __constant_compute_lowest<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto lowest(dynamic<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET           = __element_type_of_t<_Arg>;
  _ET __static_lowest = __wrapper_static_lowest<_ET, _StaticBounds>();
  return __static_lowest > __arg.__runtime_bounds_.lowest ? __static_lowest : __arg.__runtime_bounds_.lowest;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto lowest(deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET           = __element_type_of_t<_Arg>;
  _ET __static_lowest = __wrapper_static_lowest<_ET, _StaticBounds>();
  return __static_lowest > __arg.__runtime_bounds_.lowest ? __static_lowest : __arg.__runtime_bounds_.lowest;
}

template <class _Tp, ::cuda::std::enable_if_t<!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>, int> = 0>
[[nodiscard]] _CCCL_API constexpr auto max(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::max();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto max(constant<_Value>) noexcept
{
  return __constant_compute_max<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto max(dynamic<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET        = __element_type_of_t<_Arg>;
  _ET __static_max = __wrapper_static_max<_ET, _StaticBounds>();
  return __static_max < __arg.__runtime_bounds_.max ? __static_max : __arg.__runtime_bounds_.max;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto max(deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET        = __element_type_of_t<_Arg>;
  _ET __static_max = __wrapper_static_max<_ET, _StaticBounds>();
  return __static_max < __arg.__runtime_bounds_.max ? __static_max : __arg.__runtime_bounds_.max;
}

_CCCL_END_NAMESPACE_CUDA_ARGUMENT

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_H
