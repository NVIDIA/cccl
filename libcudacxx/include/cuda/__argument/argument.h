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
// constant
// =====================================================================

// Non-sequence wrappers intentionally do not reject types with a distinct element type.
// A pointer or iterator can represent either a single value or a sequence; the wrapper
// spelling carries that intent.

//! @brief Wraps a compile-time constant argument value.
template <auto _Value>
struct constant
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
struct constant_sequence
{
  using value_type     = ::cuda::std::remove_cvref_t<decltype(_Value)>;
  using __element_type = __element_type_of_t<value_type>;

  static_assert(__is_sequence_v<value_type>, "The value type of __constant_sequence must be a sequence");

  [[nodiscard]] _CCCL_API static constexpr value_type value() noexcept
  {
    return _Value;
  }
};

// __assert_in_range
// =====================================================================

template <class _To, class _From>
_CCCL_API constexpr void __assert_in_range([[maybe_unused]] _From __val) noexcept
{
  if constexpr (::cuda::std::__cccl_is_cv_integer_v<_To> && ::cuda::std::__cccl_is_cv_integer_v<_From>)
  {
    _CCCL_ASSERT(::cuda::std::in_range<::cuda::std::remove_cv_t<_To>>(__val),
                 "runtime bound value overflows the element type");
  }
}

template <class _To, class _From>
[[nodiscard]] _CCCL_API constexpr _To __runtime_bound_cast(_From __val) noexcept
{
  __assert_in_range<_To>(__val);
  return static_cast<_To>(__val);
}

template <class _To, auto _Value>
_CCCL_API constexpr bool __static_bound_in_range() noexcept
{
  using _RawTo   = ::cuda::std::remove_cv_t<_To>;
  using _RawFrom = ::cuda::std::remove_cv_t<decltype(_Value)>;

  if constexpr (::cuda::std::__cccl_is_integer_v<_RawTo> && ::cuda::std::__cccl_is_integer_v<_RawFrom>)
  {
    return ::cuda::std::in_range<_RawTo>(_Value);
  }
  else if constexpr (::cuda::std::is_arithmetic_v<_RawTo> && ::cuda::std::is_arithmetic_v<_RawFrom>)
  {
    return static_cast<_RawFrom>(static_cast<_RawTo>(_Value)) == _Value;
  }
  else
  {
    return true;
  }
}

template <class _ElementType, class _StaticBounds>
inline constexpr bool __valid_static_bounds_v = true;

template <class _ElementType, auto _Lowest, auto _Highest>
inline constexpr bool __valid_static_bounds_v<_ElementType, __static_bounds<_Lowest, _Highest>> =
  __static_bound_in_range<_ElementType, _Lowest>() && __static_bound_in_range<_ElementType, _Highest>();

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_lowest() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return ::cuda::std::numeric_limits<_ElementType>::lowest();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::lower());
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __wrapper_static_highest() noexcept
{
  if constexpr (::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    return (::cuda::std::numeric_limits<_ElementType>::max)();
  }
  else
  {
    return static_cast<_ElementType>(_StaticBounds::upper());
  }
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __effective_lowest(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  auto __static_lowest = __wrapper_static_lowest<_ElementType, _StaticBounds>();
  return __static_lowest > __runtime_bounds.lower() ? __static_lowest : __runtime_bounds.lower();
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr _ElementType __effective_highest(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  auto __static_highest = __wrapper_static_highest<_ElementType, _StaticBounds>();
  return __static_highest < __runtime_bounds.upper() ? __static_highest : __runtime_bounds.upper();
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr bool __has_bounds_intersection(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  return __effective_lowest<_ElementType, _StaticBounds>(__runtime_bounds)
      <= __effective_highest<_ElementType, _StaticBounds>(__runtime_bounds);
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr void __validate_bounds_intersection(__runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  static_assert(__valid_static_bounds_v<_ElementType, _StaticBounds>,
                "static argument bounds cannot be represented by the element type");
  _CCCL_VERIFY((__has_bounds_intersection<_ElementType, _StaticBounds>(__runtime_bounds)),
               "static and runtime argument bounds do not intersect");
}

template <class _ElementType, class _StaticBounds>
_CCCL_API constexpr void __validate_static_element_bounds([[maybe_unused]] const _ElementType& __val) noexcept
{
  if constexpr (!::cuda::std::is_same_v<_StaticBounds, __no_bounds>)
  {
    _CCCL_ASSERT((__val >= __wrapper_static_lowest<_ElementType, _StaticBounds>()),
                 "immediate argument value is below static lowest bound");
    _CCCL_ASSERT((__val <= __wrapper_static_highest<_ElementType, _StaticBounds>()),
                 "immediate argument value is above static highest bound");
  }
}

template <class _ElementType>
_CCCL_API constexpr void __validate_runtime_element_bounds(
  [[maybe_unused]] const _ElementType& __val, [[maybe_unused]] __runtime_bounds<_ElementType> __runtime_bounds) noexcept
{
  _CCCL_ASSERT((__val >= __runtime_bounds.lower()), "immediate argument value is below runtime lower bound");
  _CCCL_ASSERT((__val <= __runtime_bounds.upper()), "immediate argument value is above runtime upper bound");
}

// =====================================================================
// immediate
// =====================================================================

//! @brief Wraps a runtime argument value with optional bounds.
//!
//! The value is host-accessible at API call time.
template <class _Arg, class _StaticBounds = __no_bounds>
struct immediate
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
  _CCCL_API constexpr immediate(_Arg __arg) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_value();
  }

  _CCCL_API constexpr immediate(_Arg __arg, _StaticBounds) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_value();
  }
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE immediate(_Arg, __static_bounds<_Lowest, _Highest>)
  -> immediate<_Arg, __static_bounds<_Lowest, _Highest>>;

#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// immediate_sequence
// =====================================================================

//! @brief Wraps a runtime argument sequence with optional bounds.
template <class _Arg, class _StaticBounds = __no_bounds>
struct immediate_sequence
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
  _CCCL_API constexpr immediate_sequence(_Arg __arg) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds();
    __validate_value();
  }

  _CCCL_API constexpr immediate_sequence(_Arg __arg, _StaticBounds) noexcept
      : __arg_{::cuda::std::move(__arg)}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr immediate_sequence(_Arg __arg, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr immediate_sequence(_Arg __arg, _StaticBounds, __runtime_bounds<_BoundsTp> __rb) noexcept
      : __arg_{::cuda::std::move(__arg)}
      , __runtime_bounds_{__runtime_bound_cast<__element_type>(__rb.lower()),
                          __runtime_bound_cast<__element_type>(__rb.upper())}
  {
    __validate_bounds();
    __validate_value();
  }

  template <class _BoundsTp>
  _CCCL_API constexpr immediate_sequence(_Arg __arg, __runtime_bounds<_BoundsTp> __rb, _StaticBounds __sb) noexcept
      : immediate_sequence(::cuda::std::move(__arg), __sb, __rb)
  {}
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE immediate_sequence(_Arg, __static_bounds<_Lowest, _Highest>)
  -> immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE immediate_sequence(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE immediate_sequence(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> immediate_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __deferred_base / deferred / deferred_sequence
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
struct deferred : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE deferred(_Arg) -> deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE deferred(_Arg, __static_bounds<_Lowest, _Highest>)
  -> deferred<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE deferred(_Arg, __runtime_bounds<_Tp>) -> deferred<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE deferred(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> deferred<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE deferred(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> deferred<_Arg, __static_bounds<_Lowest, _Highest>>;
#endif // _CCCL_DOXYGEN_INVOKED

//! @brief Wraps a reference to a sequence of values that is potentially not available at API call time but will be
//! available by the time the argument is consumed in stream order.
template <class _Arg, class _StaticBounds = __no_bounds>
struct deferred_sequence : __deferred_base<_Arg, _StaticBounds>
{
  using __deferred_base<_Arg, _StaticBounds>::__deferred_base;
};

#ifndef _CCCL_DOXYGEN_INVOKED
template <class _Arg>
_CCCL_HOST_DEVICE deferred_sequence(_Arg) -> deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE deferred_sequence(_Arg, __static_bounds<_Lowest, _Highest>)
  -> deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp>
_CCCL_HOST_DEVICE deferred_sequence(_Arg, __runtime_bounds<_Tp>) -> deferred_sequence<_Arg>;

template <class _Arg, auto _Lowest, auto _Highest, class _Tp>
_CCCL_HOST_DEVICE deferred_sequence(_Arg, __static_bounds<_Lowest, _Highest>, __runtime_bounds<_Tp>)
  -> deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;

template <class _Arg, class _Tp, auto _Lowest, auto _Highest>
_CCCL_HOST_DEVICE deferred_sequence(_Arg, __runtime_bounds<_Tp>, __static_bounds<_Lowest, _Highest>)
  -> deferred_sequence<_Arg, __static_bounds<_Lowest, _Highest>>;
#endif // _CCCL_DOXYGEN_INVOKED

// =====================================================================
// __unwrap
// =====================================================================

template <class _Tp>
inline constexpr bool __is_wrapper_v = false;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<immediate<_Arg, _StaticBounds>> = true;
template <auto _Value>
inline constexpr bool __is_wrapper_v<constant<_Value>> = true;
template <auto _Value>
inline constexpr bool __is_wrapper_v<constant_sequence<_Value>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<immediate_sequence<_Arg, _StaticBounds>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<deferred<_Arg, _StaticBounds>> = true;
template <class _Arg, class _StaticBounds>
inline constexpr bool __is_wrapper_v<deferred_sequence<_Arg, _StaticBounds>> = true;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!__is_wrapper_v<::cuda::std::remove_cvref_t<_Tp>>) )
[[nodiscard]] _CCCL_API constexpr _Tp&& __unwrap(_Tp&& __arg) noexcept
{
  return ::cuda::std::forward<_Tp>(__arg);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(immediate<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const immediate<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(immediate<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::remove_cvref_t<decltype(_Value)>
__unwrap(const constant<_Value>&) noexcept
{
  return _Value;
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr ::cuda::std::remove_cvref_t<decltype(_Value)>
__unwrap(const constant_sequence<_Value>&) noexcept
{
  return _Value;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(immediate_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const immediate_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(immediate_sequence<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(deferred<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const deferred<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(deferred<_Arg, _StaticBounds>&& __arg) noexcept
{
  return ::cuda::std::move(__arg.__arg_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg& __unwrap(deferred_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr const _Arg& __unwrap(const deferred_sequence<_Arg, _StaticBounds>& __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr _Arg __unwrap(deferred_sequence<_Arg, _StaticBounds>&& __arg) noexcept
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
struct __traits_impl<constant<_Value>>
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
struct __traits_impl<immediate<_Arg, _StaticBounds>>
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
struct __traits_impl<constant_sequence<_Value>>
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
struct __traits_impl<immediate_sequence<_Arg, _StaticBounds>>
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
struct __traits_impl<deferred<_Arg, _StaticBounds>>
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
struct __traits_impl<deferred_sequence<_Arg, _StaticBounds>>
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
[[nodiscard]] _CCCL_API constexpr auto __lowest_(constant<_Value>) noexcept
{
  return __constant_compute_lowest<_Value>();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(constant_sequence<_Value>) noexcept
{
  return __constant_sequence_compute_lowest<_Value>();
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(immediate<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(immediate_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_lowest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_lowest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __lowest_(deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
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
[[nodiscard]] _CCCL_API constexpr auto __highest_(constant<_Value>) noexcept
{
  return __constant_compute_highest<_Value>();
}

template <auto _Value>
[[nodiscard]] _CCCL_API constexpr auto __highest_(constant_sequence<_Value>) noexcept
{
  return __constant_sequence_compute_highest<_Value>();
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(immediate<_Arg, _StaticBounds> __arg) noexcept
{
  return __arg.__arg_;
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(immediate_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(deferred<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

template <class _Arg, class _StaticBounds>
[[nodiscard]] _CCCL_API constexpr auto __highest_(deferred_sequence<_Arg, _StaticBounds> __arg) noexcept
{
  using _ET = __element_type_of_t<_Arg>;
  __validate_bounds_intersection<_ET, _StaticBounds>(__arg.__runtime_bounds_);
  return __effective_highest<_ET, _StaticBounds>(__arg.__runtime_bounds_);
}

_CCCL_END_NAMESPACE_CUDA_ARGUMENT

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ARGUMENT_ARGUMENT_H
