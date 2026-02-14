// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_TIME_POINT_H
#define _CUDA_STD___CHRONO_TIME_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/__compare/ordering.h>
#  include <cuda/std/__compare/three_way_comparable.h>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
template <class _Clock, class _Duration = typename _Clock::duration>
class _CCCL_TYPE_VISIBILITY_DEFAULT time_point
{
  static_assert(__is_cuda_std_duration_v<_Duration>,
                "Second template parameter of time_point must be a std::chrono::duration");

public:
  using clock    = _Clock;
  using duration = _Duration;
  using rep      = typename duration::rep;
  using period   = typename duration::period;

private:
  duration __d_;

public:
  _CCCL_API constexpr time_point()
      : __d_(duration::zero())
  {}
  _CCCL_API constexpr explicit time_point(const duration& __d)
      : __d_(__d)
  {}

  // conversions
  _CCCL_TEMPLATE(class _Duration2)
  _CCCL_REQUIRES(is_convertible_v<_Duration2, duration>)
  _CCCL_API constexpr time_point(const time_point<clock, _Duration2>& __t)
      : __d_(__t.time_since_epoch())
  {}

  // observer

  _CCCL_API constexpr duration time_since_epoch() const
  {
    return __d_;
  }

  // operations
  _CCCL_API constexpr time_point& operator++()
  {
    ++__d_;
    return *this;
  }
  _CCCL_API constexpr time_point operator++(int)
  {
    return time_point{__d_++};
  }
  _CCCL_API constexpr time_point& operator--()
  {
    --__d_;
    return *this;
  }
  _CCCL_API constexpr time_point operator--(int)
  {
    return time_point{__d_--};
  }

  // arithmetic

  _CCCL_API constexpr time_point& operator+=(const duration& __d)
  {
    __d_ += __d;
    return *this;
  }
  _CCCL_API constexpr time_point& operator-=(const duration& __d)
  {
    __d_ -= __d;
    return *this;
  }

  // special values

  [[nodiscard]] _CCCL_API static constexpr time_point min() noexcept
  {
    return time_point(duration::min());
  }
  [[nodiscard]] _CCCL_API static constexpr time_point max() noexcept
  {
    return time_point(duration::max());
  }

  // Arithmetics

  template <class _Rep2, class _Period2>
  [[nodiscard]]
  _CCCL_API friend constexpr time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>
  operator+(const time_point& __lhs, const ::cuda::std::chrono::duration<_Rep2, _Period2>& __rhs)
  {
    using _Ret = time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>;
    return _Ret{__lhs.time_since_epoch() + __rhs};
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]]
  _CCCL_API friend constexpr time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>
  operator+(const ::cuda::std::chrono::duration<_Rep2, _Period2>& __lhs, const time_point& __rhs)
  {
    using _Ret = time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>;
    return _Ret{__lhs + __rhs.time_since_epoch()};
  }

  template <class _Rep2, class _Period2>
  [[nodiscard]]
  _CCCL_API friend constexpr time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>
  operator-(const time_point& __lhs, const ::cuda::std::chrono::duration<_Rep2, _Period2>& __rhs)
  {
    using _Ret = time_point<_Clock, common_type_t<_Duration, ::cuda::std::chrono::duration<_Rep2, _Period2>>>;
    return _Ret(__lhs.time_since_epoch() - __rhs);
  }

  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr common_type_t<_Duration, _Duration2>
  operator-(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() - __rhs.time_since_epoch();
  }

  // Comparisons

  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() == __rhs.time_since_epoch();
  }

#if _CCCL_STD_VER <= 2017
  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() != __rhs.time_since_epoch();
  }
#endif // _CCCL_STD_VER <= 2017

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Duration2)
  _CCCL_REQUIRES(three_way_comparable_with<_Duration, _Duration2>)
  [[nodiscard]] _CCCL_API friend constexpr auto
  operator>=(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() <=> __rhs.time_since_epoch();
  }
#else // ^^^ _LIBCUDACXX_HAS_SPACESHIP_OPERATOR() ^^^ / vvv !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() vvv
  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() < __rhs.time_since_epoch();
  }
  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() > __rhs.time_since_epoch();
  }

  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator<=(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() <= __rhs.time_since_epoch();
  }

  template <class _Duration2>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator>=(const time_point& __lhs, const time_point<_Clock, _Duration2>& __rhs)
  {
    return __lhs.time_since_epoch() >= __rhs.time_since_epoch();
  }
#endif // !_LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
};
} // namespace chrono

template <class _Clock, class _Duration1, class _Duration2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
common_type<chrono::time_point<_Clock, _Duration1>, chrono::time_point<_Clock, _Duration2>>
{
  using type = chrono::time_point<_Clock, typename common_type<_Duration1, _Duration2>::type>;
};

namespace chrono
{
template <class _ToDuration, class _Clock, class _Duration>
[[nodiscard]] _CCCL_API constexpr time_point<_Clock, _ToDuration>
time_point_cast(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>(::cuda::std::chrono::duration_cast<_ToDuration>(__t.time_since_epoch()));
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr time_point<_Clock, _ToDuration> floor(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{::cuda::std::chrono::floor<_ToDuration>(__t.time_since_epoch())};
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr time_point<_Clock, _ToDuration> ceil(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{::cuda::std::chrono::ceil<_ToDuration>(__t.time_since_epoch())};
}

_CCCL_TEMPLATE(class _ToDuration, class _Clock, class _Duration)
_CCCL_REQUIRES(__is_cuda_std_duration_v<_ToDuration>)
[[nodiscard]] _CCCL_API constexpr time_point<_Clock, _ToDuration> round(const time_point<_Clock, _Duration>& __t)
{
  return time_point<_Clock, _ToDuration>{::cuda::std::chrono::round<_ToDuration>(__t.time_since_epoch())};
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_TIME_POINT_H
