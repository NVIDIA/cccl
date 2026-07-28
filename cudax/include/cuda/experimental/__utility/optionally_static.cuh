//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_OPTIONALLY_STATIC
#define _CUDAX__UTILITY_OPTIONALLY_STATIC

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <limits>

namespace cuda::experimental
{
template <typename _Vp>
constexpr _Vp __get_reserved_default()
{
  return ::std::is_floating_point_v<_Vp> ? -::std::numeric_limits<_Vp>::max()
       : ::std::is_unsigned_v<_Vp>
         ? ::std::numeric_limits<_Vp>::max()
         : ::std::numeric_limits<_Vp>::min();
}

template <auto _Vp>
using __copy_type_t = decltype(_Vp);

//! @brief A value that can be either a compile-time constant or a runtime value.
//!
//! When `_StaticV != _Reserved`, the value is known at compile time and occupies
//! no storage. When `_StaticV == _Reserved`, the value is dynamic and stored in
//! the object.
//!
//! All arithmetic and comparison operators are supported when the underlying
//! type supports them.
//!
//! @tparam _StaticV The static value. Equal to `_Reserved` for dynamic values.
//! @tparam _Reserved A sentinel indicating the value is dynamic.
template <auto _StaticV, __copy_type_t<_StaticV> _Reserved = __get_reserved_default<__copy_type_t<_StaticV>>()>
class optionally_static
{
public:
  using type = decltype(_StaticV);

  static constexpr bool is_static = _StaticV != _Reserved;

  static constexpr auto reserved_v = _Reserved;

  constexpr optionally_static()                                    = default;
  constexpr optionally_static(const optionally_static&)            = default;
  constexpr optionally_static& operator=(const optionally_static&) = default;

  //! @brief Construct a dynamic value. Only valid when `_StaticV == _Reserved`.
  //! @param[in] __dynamic_value The runtime value.
  constexpr optionally_static(type __dynamic_value)
      : __payload(__dynamic_value)
  {}

  //! @brief Retrieve the stored value (static or dynamic).
  //! @return The stored value.
  constexpr type get() const
  {
    if constexpr (is_static)
    {
      return _StaticV;
    }
    else
    {
      return __payload;
    }
  }

  //! @brief Implicit conversion to the underlying type.
  constexpr operator type() const
  {
    return get();
  }

  //! @brief Retrieve a mutable reference to the stored dynamic value.
  //! @return Reference to the dynamic payload.
  constexpr type& get_ref()
  {
    return __payload;
  }

  optionally_static& operator++()
  {
    ++get_ref();
    return *this;
  }

  optionally_static operator++(int)
  {
    auto __copy = *this;
    ++*this;
    return __copy;
  }

  optionally_static& operator--()
  {
    --get_ref();
    return *this;
  }

  optionally_static operator--(int)
  {
    auto __copy = *this;
    --*this;
    return __copy;
  }

  optionally_static operator+() const
  {
    return *this;
  }

  auto operator-() const
  {
    if constexpr (!is_static)
    {
      return -get();
    }
    else if constexpr (-_StaticV == _Reserved)
    {
      return _Reserved;
    }
    else
    {
      return optionally_static<-_StaticV, _Reserved>();
    }
  }

private:
  struct __nonesuch
  {};
  using __state_t                           = ::cuda::std::conditional_t<is_static, __nonesuch, type>;
  [[no_unique_address]] __state_t __payload = __state_t();
};

#ifndef _CCCL_DOXYGEN_INVOKED

// clang-format off
#  define _CUDAX_OPTIONALLY_STATIC_BINARY_OP(op)                                                            \
    template <auto _V1, auto _V2, auto _R>                                                                  \
    constexpr auto operator op(const optionally_static<_V1, _R>& __lhs,                                     \
                               const optionally_static<_V2, _R>& __rhs)                                     \
    {                                                                                                       \
      if constexpr (!::std::remove_reference_t<decltype(__lhs)>::is_static                                  \
                    || !::std::remove_reference_t<decltype(__rhs)>::is_static)                               \
      {                                                                                                     \
        return __lhs.get() op __rhs.get();                                                                  \
      }                                                                                                     \
      else if constexpr ((_V1 op _V2) == _R)                                                                \
      {                                                                                                     \
        return _R;                                                                                          \
      }                                                                                                     \
      else                                                                                                  \
      {                                                                                                     \
        return optionally_static<(_V1 op _V2), _R>();                                                       \
      }                                                                                                     \
    }                                                                                                       \
    template <auto _V, auto _R, typename _Tp>                                                               \
    constexpr auto operator op(const optionally_static<_V, _R>& __lhs, const _Tp& __rhs)                    \
    {                                                                                                       \
      return __lhs.get() op __rhs;                                                                          \
    }                                                                                                       \
    template <auto _V, auto _R, typename _Tp>                                                               \
    constexpr auto operator op(const _Tp& __lhs, const optionally_static<_V, _R>& __rhs)                    \
    {                                                                                                       \
      return __lhs op __rhs.get();                                                                          \
    }                                                                                                       \
    template <auto _V2, auto _R>                                                                            \
    constexpr auto& operator op##=(optionally_static<_R, _R>& __lhs,                                        \
                                   const optionally_static<_V2, _R>& __rhs)                                 \
    {                                                                                                       \
      return __lhs.get_ref() op## = __rhs.get();                                                            \
    }                                                                                                       \
    template <auto _R, typename _Tp>                                                                        \
    constexpr auto& operator op##=(optionally_static<_R, _R>& __lhs, const _Tp & __rhs)                     \
    {                                                                                                       \
      return __lhs.get_ref() op## = __rhs;                                                                  \
    }

_CUDAX_OPTIONALLY_STATIC_BINARY_OP(+)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(-)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(*)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(/)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(%)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(&)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(|)
_CUDAX_OPTIONALLY_STATIC_BINARY_OP(^)

#  undef _CUDAX_OPTIONALLY_STATIC_BINARY_OP

#  define _CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(op)                                                              \
    template <auto _V1, auto _V2, auto _R>                                                                       \
    constexpr bool operator op(const optionally_static<_V1, _R>& __lhs,                                          \
                               const optionally_static<_V2, _R>& __rhs)                                          \
    {                                                                                                             \
      return __lhs.get() op __rhs.get();                                                                          \
    }                                                                                                             \
    template <auto _V, auto _R, typename _Tp,                                                                    \
              typename = ::std::enable_if_t<!::std::is_same_v<_Tp, optionally_static<_V, _R>>>>                   \
    constexpr bool operator op(const optionally_static<_V, _R>& __lhs, const _Tp& __rhs)                         \
    {                                                                                                             \
      return __lhs.get() op __rhs;                                                                                \
    }                                                                                                             \
    template <auto _V, auto _R, typename _Tp,                                                                    \
              typename = ::std::enable_if_t<!::std::is_same_v<_Tp, optionally_static<_V, _R>>>>                   \
    constexpr bool operator op(const _Tp& __lhs, const optionally_static<_V, _R>& __rhs)                         \
    {                                                                                                             \
      return __lhs op __rhs.get();                                                                                \
    }

_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(==)
_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(!=)
_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(<)
_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(>)
_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(<=)
_CUDAX_OPTIONALLY_STATIC_COMPARISON_OP(>=)

#  undef _CUDAX_OPTIONALLY_STATIC_COMPARISON_OP
// clang-format on

#endif // _CCCL_DOXYGEN_INVOKED
} // namespace cuda::experimental

#endif // _CUDAX__UTILITY_OPTIONALLY_STATIC
