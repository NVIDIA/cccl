// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_DURATION_H
#define _CUDA_STD___CHRONO_DURATION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/limits>
#include <cuda/std/ratio>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
template <class _Rep, class _Period = ratio<1>>
class _CCCL_TYPE_VISIBILITY_DEFAULT duration;

template <class _Tp>
inline constexpr bool __is_cuda_std_duration_v = false;

template <class _Rep, class _Period>
inline constexpr bool __is_cuda_std_duration_v<duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_cuda_std_duration_v<const duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_cuda_std_duration_v<volatile duration<_Rep, _Period>> = true;

template <class _Rep, class _Period>
inline constexpr bool __is_cuda_std_duration_v<const volatile duration<_Rep, _Period>> = true;
} // namespace chrono

template <class _Rep1, class _Period1, class _Rep2, class _Period2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
common_type<::cuda::std::chrono::duration<_Rep1, _Period1>, ::cuda::std::chrono::duration<_Rep2, _Period2>>
{
  using type =
    ::cuda::std::chrono::duration<common_type_t<_Rep1, _Rep2>, typename __ratio_gcd<_Period1, _Period2>::type>;
};

namespace chrono
{
// duration_cast

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr _ToDuration duration_cast(const duration<_Rep, _Period>& __fd)
{
  using _FromDuration = duration<_Rep, _Period>;
  using _CPeriod      = typename ratio_divide<_Period, typename _ToDuration::period>::type;
  if constexpr (_CPeriod::num == 1 && _CPeriod::den == 1)
  {
    return _ToDuration{static_cast<typename _ToDuration::rep>(__fd.count())};
  }
  else if constexpr (_CPeriod::num == 1 && _CPeriod::den != 1)
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration{
      static_cast<typename _ToDuration::rep>(static_cast<_Ct>(__fd.count()) / static_cast<_Ct>(_CPeriod::den))};
  }
  else if constexpr (_CPeriod::num != 1 && _CPeriod::den == 1)
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration{
      static_cast<typename _ToDuration::rep>(static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_CPeriod::num))};
  }
  else // _CPeriod::num != 1 && _CPeriod::den != 1
  {
    using _Ct = common_type_t<typename _ToDuration::rep, typename _FromDuration::rep, intmax_t>;
    return _ToDuration{static_cast<typename _ToDuration::rep>(
      static_cast<_Ct>(__fd.count()) * static_cast<_Ct>(_CPeriod::num) / static_cast<_Ct>(_CPeriod::den))};
  }
}

template <class _Rep>
struct _CCCL_TYPE_VISIBILITY_DEFAULT treat_as_floating_point : is_floating_point<_Rep>
{};

template <class _Rep>
inline constexpr bool treat_as_floating_point_v = is_floating_point_v<_Rep>;

template <class _Rep>
struct _CCCL_TYPE_VISIBILITY_DEFAULT duration_values
{
public:
  [[nodiscard]] _CCCL_API static constexpr _Rep zero() noexcept
  {
    return _Rep{0};
  }
  [[nodiscard]] _CCCL_API static constexpr _Rep max() noexcept
  {
    return numeric_limits<_Rep>::max();
  }
  [[nodiscard]] _CCCL_API static constexpr _Rep min() noexcept
  {
    return numeric_limits<_Rep>::lowest();
  }
};

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr _ToDuration floor(const duration<_Rep, _Period>& __d)
{
  _ToDuration __t = ::cuda::std::chrono::duration_cast<_ToDuration>(__d);
  if (__t > __d)
  {
    __t = __t - _ToDuration{1};
  }
  return __t;
}

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr _ToDuration ceil(const duration<_Rep, _Period>& __d)
{
  _ToDuration __t = ::cuda::std::chrono::duration_cast<_ToDuration>(__d);
  if (__t < __d)
  {
    __t = __t + _ToDuration{1};
  }
  return __t;
}

_CCCL_TEMPLATE(class _ToDuration, class _Rep, class _Period)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr _ToDuration round(const duration<_Rep, _Period>& __d)
{
  _ToDuration __lower = ::cuda::std::chrono::floor<_ToDuration>(__d);
  _ToDuration __upper = __lower + _ToDuration{1};
  auto __lowerDiff    = __d - __lower;
  auto __upperDiff    = __upper - __d;
  if (__lowerDiff < __upperDiff)
  {
    return __lower;
  }
  if (__lowerDiff > __upperDiff)
  {
    return __upper;
  }
  return __lower.count() & 1 ? __upper : __lower;
}

_CCCL_TEMPLATE(class _Rep, class _Period)
_CCCL_REQUIRES(numeric_limits<_Rep>::is_signed)
[[nodiscard]] _CCCL_API constexpr duration<_Rep, _Period> abs(duration<_Rep, _Period> __d)
{
  return __d >= __d.zero() ? +__d : -__d;
}

// duration

template <class _Rep, class _Period>
class _CCCL_TYPE_VISIBILITY_DEFAULT duration
{
  static_assert(!__is_cuda_std_duration_v<_Rep>, "A duration representation can not be a duration");
  static_assert(__is_cuda_std_ratio_v<_Period>, "Second template parameter of duration must be a std::ratio");
  static_assert(_Period::num > 0, "duration period must be positive");

  template <class _R1, class _R2>
  struct __no_overflow
  {
  private:
    static constexpr intmax_t __gcd_n1_n2 = __static_gcd<_R1::num, _R2::num>::value;
    static constexpr intmax_t __gcd_d1_d2 = __static_gcd<_R1::den, _R2::den>::value;
    static constexpr intmax_t __n1        = _R1::num / __gcd_n1_n2;
    static constexpr intmax_t __d1        = _R1::den / __gcd_d1_d2;
    static constexpr intmax_t __n2        = _R2::num / __gcd_n1_n2;
    static constexpr intmax_t __d2        = _R2::den / __gcd_d1_d2;
    static constexpr intmax_t max         = -((intmax_t(1) << (sizeof(intmax_t) * CHAR_BIT - 1)) + 1);

    template <intmax_t _Xp, intmax_t _Yp, bool __overflow>
    struct __mul // __overflow == false
    {
      static constexpr intmax_t value = _Xp * _Yp;
    };

    template <intmax_t _Xp, intmax_t _Yp>
    struct __mul<_Xp, _Yp, true>
    {
      static constexpr intmax_t value = 1;
    };

  public:
    static constexpr bool value = (__n1 <= max / __d2) && (__n2 <= max / __d1);
    using type                  = ratio<__mul<__n1, __d2, !value>::value, __mul<__n2, __d1, !value>::value>;
  };

public:
  using rep    = _Rep;
  using period = typename _Period::type;

private:
  rep __rep_;

public:
  _CCCL_HIDE_FROM_ABI constexpr duration() = default;

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES(
    is_convertible_v<const _Rep2&, rep> _CCCL_AND(treat_as_floating_point_v<rep> || !treat_as_floating_point_v<_Rep2>))
  _CCCL_API constexpr explicit duration(const _Rep2& __r)
      : __rep_(static_cast<rep>(__r))
  {}

  // conversions
  _CCCL_TEMPLATE(class _Rep2, class _Period2)
  _CCCL_REQUIRES(__no_overflow<_Period2, period>::value _CCCL_AND(
    treat_as_floating_point_v<rep>
    || (__no_overflow<_Period2, period>::type::den == 1 && !treat_as_floating_point_v<_Rep2>) ))
  _CCCL_API constexpr duration(const duration<_Rep2, _Period2>& __d)
      : __rep_(::cuda::std::chrono::duration_cast<duration>(__d).count())
  {}

  // observer

  [[nodiscard]] _CCCL_API constexpr rep count() const
  {
    return __rep_;
  }

  // arithmetic

  _CCCL_API constexpr common_type_t<duration> operator+() const
  {
    return common_type_t<duration>(*this);
  }
  _CCCL_API constexpr common_type_t<duration> operator-() const
  {
    return common_type_t<duration>(-__rep_);
  }
  _CCCL_API constexpr duration& operator++()
  {
    ++__rep_;
    return *this;
  }
  _CCCL_API constexpr duration operator++(int)
  {
    return duration{__rep_++};
  }
  _CCCL_API constexpr duration& operator--()
  {
    --__rep_;
    return *this;
  }
  _CCCL_API constexpr duration operator--(int)
  {
    return duration{__rep_--};
  }

  _CCCL_API constexpr duration& operator+=(const duration& __d)
  {
    __rep_ += __d.count();
    return *this;
  }
  _CCCL_API constexpr duration& operator-=(const duration& __d)
  {
    __rep_ -= __d.count();
    return *this;
  }

  _CCCL_API constexpr duration& operator*=(const rep& rhs)
  {
    __rep_ *= rhs;
    return *this;
  }
  _CCCL_API constexpr duration& operator/=(const rep& rhs)
  {
    __rep_ /= rhs;
    return *this;
  }
  _CCCL_API constexpr duration& operator%=(const rep& rhs)
  {
    __rep_ %= rhs;
    return *this;
  }
  _CCCL_API constexpr duration& operator%=(const duration& rhs)
  {
    __rep_ %= rhs.count();
    return *this;
  }

  // special values

  [[nodiscard]] _CCCL_API static constexpr duration zero() noexcept
  {
    return duration{duration_values<rep>::zero()};
  }
  [[nodiscard]] _CCCL_API static constexpr duration min() noexcept
  {
    return duration{duration_values<rep>::min()};
  }
  [[nodiscard]] _CCCL_API static constexpr duration max() noexcept
  {
    return duration{duration_values<rep>::max()};
  }

  // Comparisons

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() == __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration, duration<_Rep2, _Period2>>;
      return _Ct(__lhs).count() == _Ct(__rhs).count();
    }
  }

#if _CCCL_STD_VER <= 2017
  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() != __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration, duration<_Rep2, _Period2>>;
      return _Ct(__lhs).count() != _Ct(__rhs).count();
    }
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Rep2, class _Period2)
  _CCCL_REQUIRES(three_way_comparable<common_type_t<_Rep1, _Rep2>>)
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator<=>(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() <=> __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration, duration<_Rep2, _Period2>>;
      return _Ct(__lhs).count() <=> _Ct(__rhs).count();
    }
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    if constexpr (is_same_v<duration, duration<_Rep2, _Period2>>)
    {
      return __lhs.count() < __rhs.count();
    }
    else
    {
      using _Ct = common_type_t<duration, duration<_Rep2, _Period2>>;
      return _Ct(__lhs).count() < _Ct(__rhs).count();
    }
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return __rhs < __lhs;
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return !(__rhs < __lhs);
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    return !(__lhs < __rhs);
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  // Arithmetic

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr common_type_t<duration, duration<_Rep2, _Period2>>
  operator+(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cd = common_type_t<duration, duration<_Rep2, _Period2>>;
    return _Cd(_Cd(__lhs).count() + _Cd(__rhs).count());
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr common_type_t<duration, duration<_Rep2, _Period2>>
  operator-(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cd = common_type_t<duration, duration<_Rep2, _Period2>>;
    return _Cd(_Cd(__lhs).count() - _Cd(__rhs).count());
  }

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES(is_convertible_v<const _Rep2, common_type_t<_Rep, _Rep2>>)
  [[nodiscard]] _CCCL_API friend constexpr duration<common_type_t<_Rep, _Rep2>, _Period>
  operator*(const duration& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() * static_cast<_Cr>(__s));
  }

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES(is_convertible_v<const _Rep2&, common_type_t<_Rep, _Rep2>>)
  [[nodiscard]] _CCCL_API friend constexpr duration<common_type_t<_Rep, _Rep2>, _Period>
  operator*(const _Rep2& __s, const duration& __d)
  {
    using _Cr = common_type_t<_Rep, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() * static_cast<_Cr>(__s));
  }

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES((!__is_cuda_std_duration_v<_Rep2>) _CCCL_AND is_convertible_v<const _Rep2&, common_type_t<_Rep, _Rep2>>)
  [[nodiscard]] _CCCL_API friend constexpr duration<common_type_t<_Rep, _Rep2>, _Period>
  operator/(const duration& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() / static_cast<_Cr>(__s));
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr common_type_t<_Rep, _Rep2>
  operator/(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Ct = common_type_t<duration, duration<_Rep2, _Period2>>;
    return _Ct(__lhs).count() / _Ct(__rhs).count();
  }

  _CCCL_TEMPLATE(class _Rep2)
  _CCCL_REQUIRES((!__is_cuda_std_duration_v<_Rep2>) _CCCL_AND is_convertible_v<const _Rep2&, common_type_t<_Rep, _Rep2>>)
  [[nodiscard]] _CCCL_API friend constexpr duration<common_type_t<_Rep, _Rep2>, _Period>
  operator%(const duration& __d, const _Rep2& __s)
  {
    using _Cr = common_type_t<_Rep, _Rep2>;
    using _Cd = duration<_Cr, _Period>;
    return _Cd(_Cd(__d).count() % static_cast<_Cr>(__s));
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]] _CCCL_API friend constexpr common_type_t<duration, duration<_Rep2, _Period2>>
  operator%(const duration& __lhs, const duration<_Rep2, _Period2>& __rhs)
  {
    using _Cr = common_type_t<_Rep, _Rep2>;
    using _Cd = common_type_t<duration, duration<_Rep2, _Period2>>;
    return _Cd(static_cast<_Cr>(_Cd(__lhs).count()) % static_cast<_Cr>(_Cd(__rhs).count()));
  }
};

using nanoseconds  = duration<long long, nano>;
using microseconds = duration<long long, micro>;
using milliseconds = duration<long long, milli>;
using seconds      = duration<long long>;
using minutes      = duration<long, ratio<60>>;
using hours        = duration<long, ratio<3600>>;

using days   = duration<int, ratio_multiply<ratio<24>, hours::period>>;
using weeks  = duration<int, ratio_multiply<ratio<7>, days::period>>;
using years  = duration<int, ratio_multiply<ratio<146097, 400>, days::period>>;
using months = duration<int, ratio_divide<years::period, ratio<12>>>;
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_DURATION_H
