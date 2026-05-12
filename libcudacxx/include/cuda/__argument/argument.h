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
// __element_type_of
// =====================================================================

template <class _Tp>
struct __element_type_of
{
  using type = _Tp;
};

template <class _Tp>
struct __element_type_of<_Tp*>
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
// __constant
// =====================================================================

//! @brief Wraps a compile-time constant argument value.
//!
//! Supports both scalar values and arrays (e.g., @c __constant<std::array{128, 256, 512}>).
//! For arrays, bounds are computed as the min/max of the elements.
template <auto _Value>
struct __constant
{
  using value_type                  = decltype(_Value);
  using __element_type              = __element_type_of_t<value_type>;
  static constexpr value_type value = _Value;
};

// =====================================================================
// __immediate
// =====================================================================

//! @brief Wraps a runtime argument value with optional bounds.
//!
//! The value is host-accessible at API call time, in contrast to deferred arguments.
//! When a scalar value is provided alongside bounds, the value is
//! validated against the bounds at construction time (debug-only).
//! Runtime bounds are only supported for collection types (spans, etc.),
//! not for single-value types.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __immediate
{
  using __element_type = __element_type_of_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  __runtime_bounds<__element_type> __runtime_bounds_;

private:
  _CCCL_API constexpr void __validate_element(__element_type __val) noexcept
  {
    if constexpr (!::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
    {
      _CCCL_ASSERT(__val >= __static_bounds_.lowest, "immediate argument value is below static lowest bound");
      _CCCL_ASSERT(__val <= __static_bounds_.max, "immediate argument value is above static max bound");
    }
    _CCCL_ASSERT(__val >= __runtime_bounds_.lowest, "immediate argument value is below runtime lowest bound");
    _CCCL_ASSERT(__val <= __runtime_bounds_.max, "immediate argument value is above runtime max bound");
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
  _CCCL_API constexpr __immediate(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Lowest, auto _Max>
  _CCCL_API constexpr __immediate(_Arg __arg, __static_bounds<_Lowest, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {
    __validate();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __immediate(_Arg __arg, __runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {
    static_assert(!::cuda::std::is_same_v<_Arg, __element_type>,
                  "runtime bounds on a single-value immediate argument are not supported; use static bounds instead");
    __validate();
  }

  template <auto _Lowest, auto _Max, class _BoundsTp>
  _CCCL_API constexpr __immediate(
    _Arg __arg, __static_bounds<_Lowest, _Max> __sb, __runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {
    static_assert(!::cuda::std::is_same_v<_Arg, __element_type>,
                  "runtime bounds on a single-value immediate argument are not supported; use static bounds instead");
    __validate();
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE __immediate(_Arg) -> __immediate<_Arg>;

template <class _Arg, auto _Lowest, auto _Max>
_CCCL_HOST_DEVICE __immediate(_Arg, __static_bounds<_Lowest, _Max>)
  -> __immediate<_Arg, __static_bounds<_Lowest, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE __immediate(_Arg, __runtime_bounds<_Tp>) -> __immediate<_Arg>;

template <class _Arg, auto _Lowest, auto _Max, class _Tp>
_CCCL_HOST_DEVICE __immediate(_Arg, __static_bounds<_Lowest, _Max>, __runtime_bounds<_Tp>)
  -> __immediate<_Arg, __static_bounds<_Lowest, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __deferred_base / __deferred_value / __deferred_sequence
// =====================================================================

//! @brief Common base for deferred argument wrappers.
//!
//! Holds a device-resident value that is not host-accessible at API call time.
//! The dispatch layer can only use the attached bounds to make host-side decisions.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred_base
{
  using __element_type = __element_type_of_t<_Arg>;

  _Arg arg;
  _CCCL_NO_UNIQUE_ADDRESS _StaticBounds __static_bounds_;
  __runtime_bounds<__element_type> __runtime_bounds_;

  _CCCL_API constexpr __deferred_base(_Arg __arg) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{}
  {}

  template <auto _Lowest, auto _Max>
  _CCCL_API constexpr __deferred_base(_Arg __arg, __static_bounds<_Lowest, _Max> __sb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{}
  {}

  template <class _BoundsTp>
  _CCCL_API constexpr __deferred_base(_Arg __arg, __runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_{}
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {}

  template <auto _Lowest, auto _Max, class _BoundsTp>
  _CCCL_API constexpr __deferred_base(
    _Arg __arg, __static_bounds<_Lowest, _Max> __sb, __runtime_bounds<_BoundsTp> __rb) noexcept
      : arg(__arg)
      , __static_bounds_(__sb)
      , __runtime_bounds_{static_cast<__element_type>(__rb.lowest), static_cast<__element_type>(__rb.max)}
  {}
};

//! @brief Wraps a single device-resident value (pointer, fancy iterator, span<T,1>, etc.).
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred_value : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE __deferred_value(_Arg) -> __deferred_value<_Arg>;

template <class _Arg, auto _Lowest, auto _Max>
_CCCL_HOST_DEVICE __deferred_value(_Arg, __static_bounds<_Lowest, _Max>)
  -> __deferred_value<_Arg, __static_bounds<_Lowest, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE __deferred_value(_Arg, __runtime_bounds<_Tp>) -> __deferred_value<_Arg>;

template <class _Arg, auto _Lowest, auto _Max, class _Tp>
_CCCL_HOST_DEVICE __deferred_value(_Arg, __static_bounds<_Lowest, _Max>, __runtime_bounds<_Tp>)
  -> __deferred_value<_Arg, __static_bounds<_Lowest, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

//! @brief Wraps a device-resident sequence of values (pointer, span, iterator, etc.).
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred_sequence : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg) -> __deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Max>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __static_bounds<_Lowest, _Max>)
  -> __deferred_sequence<_Arg, __static_bounds<_Lowest, _Max>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __runtime_bounds<_Tp>) -> __deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Max, class _Tp>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __static_bounds<_Lowest, _Max>, __runtime_bounds<_Tp>)
  -> __deferred_sequence<_Arg, __static_bounds<_Lowest, _Max>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __unwrap
// =====================================================================

template <class _Tp>
inline constexpr bool __is_wrapper_v = false;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__immediate<_Arg, _StaticBounds>> = true;
template <auto _Value>
inline constexpr bool __is_wrapper_v<__constant<_Value>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__deferred_value<_Arg, _StaticBounds>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__deferred_sequence<_Arg, _StaticBounds>> = true;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<_Tp>>>) )
[[nodiscard]] _CCCL_API constexpr _Tp&& __unwrap(_Tp&& __arg) noexcept
{
  return ::cuda::std::forward<_Tp>(__arg);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __immediate<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr decltype(_Value) __unwrap(const __constant<_Value>&) noexcept
{
  return _Value;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __deferred_value<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.arg;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __deferred_sequence<_Arg, _StaticBounds>& __arg) noexcept
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
// __traits
// =====================================================================

//! @brief Traits for argument wrappers and plain argument values.
//!
//! Models @c numeric_limits for bounds: @c lowest is the lower bound, @c max is the upper bound.
//! Use in @c if @c constexpr for compile-time dispatch based on bounds.
template <class _Tp>
struct __traits
{
  using value_type                      = _Tp;
  using element_type                    = __element_type_of_t<_Tp>;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = __is_single_value_v<_Tp>;
  static constexpr element_type lowest  = ::cuda::std::numeric_limits<element_type>::lowest();
  static constexpr element_type max     = ::cuda::std::numeric_limits<element_type>::max();
};

template <class _Arg, class _StaticBounds>
struct __traits<__immediate<_Arg, _StaticBounds>>
{
  using value_type                      = _Arg;
  using element_type                    = __element_type_of_t<_Arg>;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = __is_single_value_v<_Arg>;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type max     = __wrapper_static_max<element_type, _StaticBounds>();
};

template <auto _Value>
struct __traits<__constant<_Value>>
{
  using value_type                      = decltype(_Value);
  using element_type                    = __element_type_of_t<value_type>;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = __is_single_value_v<value_type>;
  static constexpr element_type lowest  = __constant_compute_lowest<_Value>;
  static constexpr element_type max     = __constant_compute_max<_Value>;
};

template <class _Arg, class _StaticBounds>
struct __traits<__deferred_value<_Arg, _StaticBounds>>
{
  using value_type                      = _Arg;
  using element_type                    = __element_type_of_t<_Arg>;
  static constexpr bool is_deferred     = true;
  static constexpr bool is_single_value = true;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type max     = __wrapper_static_max<element_type, _StaticBounds>();
};

template <class _Arg, class _StaticBounds>
struct __traits<__deferred_sequence<_Arg, _StaticBounds>>
{
  using value_type                      = _Arg;
  using element_type                    = __element_type_of_t<_Arg>;
  static constexpr bool is_deferred     = true;
  static constexpr bool is_single_value = false;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type max     = __wrapper_static_max<element_type, _StaticBounds>();
};

// =====================================================================
// __lowest / __max — free functions
// =====================================================================

//! @brief Returns the effective lowest bound, combining static and runtime bounds.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr auto __lowest(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::lowest();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __lowest(__constant<_Value>) noexcept
{
  return __constant_compute_lowest<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest(__immediate<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET           = __element_type_of_t<_Arg>;
  _ET __static_lowest = __wrapper_static_lowest<_ET, _StaticBounds>();
  return __static_lowest > __arg.__runtime_bounds_.lowest ? __static_lowest : __arg.__runtime_bounds_.lowest;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest(__deferred_value<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET           = __element_type_of_t<_Arg>;
  _ET __static_lowest = __wrapper_static_lowest<_ET, _StaticBounds>();
  return __static_lowest > __arg.__runtime_bounds_.lowest ? __static_lowest : __arg.__runtime_bounds_.lowest;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest(__deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET           = __element_type_of_t<_Arg>;
  _ET __static_lowest = __wrapper_static_lowest<_ET, _StaticBounds>();
  return __static_lowest > __arg.__runtime_bounds_.lowest ? __static_lowest : __arg.__runtime_bounds_.lowest;
}

//! @brief Returns the effective max bound, combining static and runtime bounds.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr auto __max(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::max();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __max(__constant<_Value>) noexcept
{
  return __constant_compute_max<_Value>;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __max(__immediate<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET        = __element_type_of_t<_Arg>;
  _ET __static_max = __wrapper_static_max<_ET, _StaticBounds>();
  return __static_max < __arg.__runtime_bounds_.max ? __static_max : __arg.__runtime_bounds_.max;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __max(__deferred_value<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET        = __element_type_of_t<_Arg>;
  _ET __static_max = __wrapper_static_max<_ET, _StaticBounds>();
  return __static_max < __arg.__runtime_bounds_.max ? __static_max : __arg.__runtime_bounds_.max;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __max(__deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET        = __element_type_of_t<_Arg>;
  _ET __static_max = __wrapper_static_max<_ET, _StaticBounds>();
  return __static_max < __arg.__runtime_bounds_.max ? __static_max : __arg.__runtime_bounds_.max;
}

_CCCL_END_NAMESPACE_CUDA_ARGUMENT

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_H
