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
#include <cuda/std/__algorithm/max_element.h>
#include <cuda/std/__algorithm/min_element.h>
#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_ARGUMENT

// =====================================================================
// __element_type_of
// =====================================================================

template <class _Tp, class = void>
struct __element_type_from_member_iterator
{
  using type = _Tp;
};

template <class _Tp>
struct __element_type_from_member_iterator<_Tp, ::cuda::std::void_t<::cuda::std::iter_value_t<typename _Tp::iterator>>>
{
  using type = ::cuda::std::iter_value_t<typename _Tp::iterator>;
};

// Fallback: element type is the type itself.
template <class _Tp, class = void>
struct __element_type_of : __element_type_from_member_iterator<_Tp>
{};

template <class _Tp>
struct __element_type_of<_Tp, ::cuda::std::void_t<::cuda::std::iter_value_t<_Tp>>>
{
  using type = ::cuda::std::iter_value_t<_Tp>;
};

template <class _Tp>
using __element_type_of_t = typename __element_type_of<::cuda::std::remove_cvref_t<_Tp>>::type;

// =====================================================================
// __is_sequence_v
// =====================================================================

template <class _Tp>
inline constexpr bool __is_sequence_v =
  (::cuda::std::is_array_v<::cuda::std::remove_cvref_t<_Tp>> || ::cuda::std::ranges::range<_Tp>)
  || ::cuda::std::__has_random_access_traversal<_Tp>;

// =====================================================================
// __constant
// =====================================================================

// Non-sequence wrappers intentionally do not reject types with a distinct element type.
// A pointer or iterator can represent either a single value or a sequence; the wrapper
// spelling carries that intent.

//! @brief Wraps a compile-time constant argument value.
template <auto _Value>
struct __constant
{
  using value_type     = ::cuda::std::remove_cvref_t<decltype(_Value)>;
  using __element_type = value_type;

  [[nodiscard]] _CCCL_API static constexpr value_type value() noexcept
  {
    return _Value;
  }
};

//! @brief Wraps a compile-time constant argument sequence.
template <auto _Value>
struct __constant_sequence
{
  using value_type     = ::cuda::std::remove_cvref_t<decltype(_Value)>;
  using __element_type = __element_type_of_t<value_type>;

  static_assert(__is_sequence_v<value_type>, "The value type of __constant_sequence must be a sequence");

  [[nodiscard]] _CCCL_API static constexpr value_type value() noexcept
  {
    return _Value;
  }
};

// =====================================================================
// __immediate
// =====================================================================

//! @brief Wraps a runtime argument value with optional bounds.
//!
//! The value is host-accessible at API call time.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __immediate
{
  using __element_type = __element_type_of_t<_Arg>;

  static_assert(__valid_static_bounds_v<__element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  _Arg __arg_;

private:
  _CCCL_API constexpr void __validate_value() const noexcept
  {
    if constexpr (::cuda::std::is_same_v<::cuda::std::remove_cvref_t<_Arg>, __element_type>
                  && ::cuda::std::is_arithmetic_v<__element_type>)
    {
      __validate_static_element_bounds<__element_type, _StaticBounds>(__arg_);
    }
  }

public:
  _CCCL_API constexpr __immediate(_Arg __arg) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_value();
  }

  _CCCL_API constexpr __immediate(_Arg __arg, _StaticBounds) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_value();
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __immediate(_Arg, __static_bounds<_Lowest, _Highest>)
  -> __immediate<_Arg, __static_bounds<_Lowest, _Highest>>;

#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __immediate_sequence
// =====================================================================

//! @brief Wraps a runtime argument sequence with optional bounds.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __immediate_sequence
{
  using __element_type = __element_type_of_t<_Arg>;

  static_assert(__is_sequence_v<_Arg>, "immediate sequence arguments must have a distinct element type");
  static_assert(__valid_static_bounds_v<__element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  _Arg __arg_;
  __runtime_bounds<__element_type> __runtime_bounds_{};

private:
  _CCCL_API constexpr void __validate_bounds() const noexcept
  {
    __validate_bounds_intersection<__element_type, _StaticBounds>(__runtime_bounds_);
  }

  _CCCL_API constexpr void __validate_element(const __element_type& __val) const noexcept
  {
    __validate_static_element_bounds<__element_type, _StaticBounds>(__val);
    __validate_runtime_element_bounds(__val, __runtime_bounds_);
  }

  _CCCL_API constexpr void __validate_value() const noexcept
  {
    if constexpr (::cuda::std::__has_random_access_traversal<_Arg>)
    { // FIXME: (miscco) This is broken. we do not know the size of the sequence
    }
    else if constexpr (__is_sequence_v<_Arg> && !::cuda::std::__has_random_access_traversal<_Arg>
                       && ::cuda::std::is_arithmetic_v<__element_type>)
    {
      for (const auto& __a : __arg_)
      {
        __validate_element(__a);
      }
    }
  }

public:
  _CCCL_API constexpr __immediate_sequence(_Arg __arg) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds();
    __validate_value();
  }

  _CCCL_API constexpr __immediate_sequence(_Arg __arg, _StaticBounds) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __immediate_sequence(_Arg __arg, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __immediate_sequence(_Arg __arg, _StaticBounds, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __immediate_sequence(_Arg __arg, __runtime_bounds<_BoundsTp> __rb, _StaticBounds __sb) noexcept
      : __immediate_sequence(::cuda::std::move(__arg), __sb, __rb)
  {}
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __immediate_sequence(_Arg, __static_bounds<_Lowest, _Highest>)
  -> __immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE __immediate_sequence(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> __immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __immediate_sequence(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> __immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __deferred_base / __deferred / __deferred_sequence
// =====================================================================

//! @brief Common base for deferred argument wrappers.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred_base
{
  using __element_type = __element_type_of_t<_Arg>;

  static_assert(__valid_static_bounds_v<__element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  _Arg __arg_;
  __runtime_bounds<__element_type> __runtime_bounds_{};

  _CCCL_API constexpr __deferred_base(_Arg __arg) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds_intersection<__element_type, _StaticBounds>(__runtime_bounds_);
  }

  _CCCL_API constexpr __deferred_base(_Arg __arg, _StaticBounds) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds_intersection<__element_type, _StaticBounds>(__runtime_bounds_);
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __deferred_base(_Arg __arg, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds_intersection<__element_type, _StaticBounds>(__runtime_bounds_);
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __deferred_base(_Arg __arg, _StaticBounds, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds_intersection<__element_type, _StaticBounds>(__runtime_bounds_);
  }

  template <class _BoundsTp>
  _CCCL_API constexpr __deferred_base(_Arg __arg, __runtime_bounds<_BoundsTp> __rb, _StaticBounds __sb) noexcept
      : __deferred_base(::cuda::std::move(__arg), __sb, __rb)
  {}
};

//! @brief Wraps a reference to a single value that is potentially not available at API call time but will be available
//! by the time the argument is consumed in stream order.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE __deferred(_Arg) -> __deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __deferred(_Arg, __static_bounds<_Lowest, _Highest>)
  -> __deferred<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE __deferred(_Arg, __runtime_bounds<_Tp>) -> __deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE __deferred(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> __deferred<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __deferred(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> __deferred<_Arg, __static_bounds<_Lowest, _Highest>>;
#endif // _CCCL_DOXYGEN_INVOKED

//! @brief Wraps a reference to a sequence of values that is potentially not available at API call time but will be
//! available by the time the argument is consumed in stream order.
template <class _Arg, class _StaticBounds = __no_bounds>
struct __deferred_sequence : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg) -> __deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __static_bounds<_Lowest, _Highest>)
  -> __deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __runtime_bounds<_Tp>) -> __deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> __deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE __deferred_sequence(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> __deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;
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
template <auto _Value>
inline constexpr bool __is_wrapper_v<__constant_sequence<_Value>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__immediate_sequence<_Arg, _StaticBounds>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__deferred<_Arg, _StaticBounds>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<__deferred_sequence<_Arg, _StaticBounds>> = true;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cvref_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr _Tp&& __unwrap(_Tp&& __arg) noexcept
{
  return ::cuda::std::forward<_Tp>(__arg);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(__immediate<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __immediate<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(__immediate<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::remove_cvref_t<decltype(_Value)>
__unwrap(const __constant<_Value>&) noexcept
{
  return _Value;
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::remove_cvref_t<decltype(_Value)>
__unwrap(const __constant_sequence<_Value>&) noexcept
{
  return _Value;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(__immediate_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __immediate_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(__immediate_sequence<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(__deferred<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __deferred<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(__deferred<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(__deferred_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const __deferred_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(__deferred_sequence<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <auto _Value>
_CCCL_API constexpr auto __constant_compute_lowest() noexcept
{
  return _Value;
}

template <auto _Value>
_CCCL_API constexpr auto __constant_compute_highest() noexcept
{
  return _Value;
}

template <auto _Value>
_CCCL_API constexpr auto __constant_sequence_compute_lowest() noexcept
{
  using _ElementType = __element_type_of_t<::cuda::std::remove_cvref_t<decltype(_Value)>>;
  auto __first       = _Value.begin();
  auto __last        = _Value.end();

  if (__first == __last)
  {
    return ::cuda::std::numeric_limits<_ElementType>::lowest();
  }
  return static_cast<_ElementType>(*::cuda::std::min_element(__first, __last));
}

template <auto _Value>
_CCCL_API constexpr auto __constant_sequence_compute_highest() noexcept
{
  using _ElementType = __element_type_of_t<::cuda::std::remove_cvref_t<decltype(_Value)>>;
  auto __first       = _Value.begin();
  auto __last        = _Value.end();

  if (__first == __last)
  {
    return (::cuda::std::numeric_limits<_ElementType>::max)();
  }
  return static_cast<_ElementType>(*::cuda::std::max_element(__first, __last));
}

// =====================================================================
// __traits
// =====================================================================

//! @brief Traits for argument wrappers and plain argument values.
//!
//! Models @c numeric_limits for bounds: @c lowest is the lower bound, @c highest is the upper bound.
//! Use in @c if @c constexpr for compile-time dispatch based on bounds.
template <class _Tp>
struct __traits_impl
{
  using value_type                      = _Tp;
  using element_type                    = __element_type_of_t<_Tp>;
  static constexpr bool is_constant     = false;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = true;
  static constexpr element_type lowest  = ::cuda::std::numeric_limits<element_type>::lowest();
  static constexpr element_type highest = (::cuda::std::numeric_limits<element_type>::max)();
};

template <auto _Value>
struct __traits_impl<__constant<_Value>>
{
  using value_type                      = ::cuda::std::remove_cvref_t<decltype(_Value)>;
  using element_type                    = value_type;
  static constexpr bool is_constant     = true;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = true;
  static constexpr element_type lowest  = __constant_compute_lowest<_Value>();
  static constexpr element_type highest = __constant_compute_highest<_Value>();
};

template <class _Arg, class _StaticBounds>
struct __traits_impl<__immediate<_Arg, _StaticBounds>>
{
  using value_type   = _Arg;
  using element_type = __element_type_of_t<_Arg>;
  static_assert(__valid_static_bounds_v<element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  static constexpr bool is_constant     = false;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = true;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type highest = __wrapper_static_highest<element_type, _StaticBounds>();
};

template <auto _Value>
struct __traits_impl<__constant_sequence<_Value>>
{
  using value_type   = ::cuda::std::remove_cvref_t<decltype(_Value)>;
  using element_type = __element_type_of_t<value_type>;
  static_assert(__is_sequence_v<value_type>, "The value type of __constant_sequence must be a sequence");
  static constexpr bool is_constant     = true;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = false;
  static constexpr element_type lowest  = __constant_sequence_compute_lowest<_Value>();
  static constexpr element_type highest = __constant_sequence_compute_highest<_Value>();
};

template <class _Arg, class _StaticBounds>
struct __traits_impl<__immediate_sequence<_Arg, _StaticBounds>>
{
  using value_type   = _Arg;
  using element_type = __element_type_of_t<_Arg>;
  static_assert(__is_sequence_v<value_type>, "immediate sequence arguments must have a distinct element type");
  static_assert(__valid_static_bounds_v<element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  static constexpr bool is_constant     = false;
  static constexpr bool is_deferred     = false;
  static constexpr bool is_single_value = false;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type highest = __wrapper_static_highest<element_type, _StaticBounds>();
};

template <class _Arg, class _StaticBounds>
struct __traits_impl<__deferred<_Arg, _StaticBounds>>
{
  using value_type   = _Arg;
  using element_type = __element_type_of_t<_Arg>;
  static_assert(__valid_static_bounds_v<element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  static constexpr bool is_constant     = false;
  static constexpr bool is_deferred     = true;
  static constexpr bool is_single_value = true;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type highest = __wrapper_static_highest<element_type, _StaticBounds>();
};

template <class _Arg, class _StaticBounds>
struct __traits_impl<__deferred_sequence<_Arg, _StaticBounds>>
{
  using value_type   = _Arg;
  using element_type = __element_type_of_t<_Arg>;
  static_assert(__valid_static_bounds_v<element_type, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");

  static constexpr bool is_constant     = false;
  static constexpr bool is_deferred     = true;
  static constexpr bool is_single_value = false;
  static constexpr element_type lowest  = __wrapper_static_lowest<element_type, _StaticBounds>();
  static constexpr element_type highest = __wrapper_static_highest<element_type, _StaticBounds>();
};

template <class _Tp>
struct __traits : __traits_impl<::cuda::std::remove_cvref_t<_Tp>>
{};

// =====================================================================
// __lowest_ / __highest_ — free functions
// =====================================================================

//! @brief Returns the effective lowest bound, combining static and runtime bounds.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr auto __lowest_(_Tp) noexcept
{
  return ::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::lowest();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__constant<_Value>) noexcept
{
  return __constant_compute_lowest<_Value>();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__constant_sequence<_Value>) noexcept
{
  return __constant_sequence_compute_lowest<_Value>();
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__immediate<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__immediate_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_lowest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_lowest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(__deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_lowest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

//! @brief Returns the effective highest bound, combining static and runtime bounds.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cv_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr auto __highest_(_Tp) noexcept
{
  return (::cuda::std::numeric_limits<__element_type_of_t<_Tp>>::max)();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__constant<_Value>) noexcept
{
  return __constant_compute_highest<_Value>();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__constant_sequence<_Value>) noexcept
{
  return __constant_sequence_compute_highest<_Value>();
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__immediate<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__immediate_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(__deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

_CCCL_END_NAMESPACE_CUDA_ARGUMENT

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_H
