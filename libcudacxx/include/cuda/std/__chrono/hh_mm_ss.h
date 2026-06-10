// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CHRONO_HH_MM_SS_H
#define _CUDA_STD___CHRONO_HH_MM_SS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__chrono/duration.h>
#include <cuda/std/__chrono/time_point.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/ratio>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace chrono
{
template <class _Duration>
class hh_mm_ss
{
private:
  static_assert(__is_cuda_std_duration_v<_Duration>, "template parameter of hh_mm_ss must be a std::chrono::duration");
  using __common_type = common_type_t<_Duration, chrono::seconds>;

  [[nodiscard]] _CCCL_API static constexpr uint64_t __pow10(unsigned __exp)
  {
    uint64_t __ret = 1;
    for (unsigned __i = 0; __i < __exp; ++__i)
    {
      __ret *= 10U;
    }
    return __ret;
  }

  [[nodiscard]] _CCCL_API static constexpr unsigned __width(uint64_t __n, uint64_t __d = 10, unsigned __w = 0)
  {
    if (__n >= 2 && __d != 0 && __w < 19)
    {
      return 1 + __width(__n, __d % __n * 10, __w + 1);
    }
    return 0;
  }

public:
  static unsigned constexpr fractional_width =
    __width(__common_type::period::den) < 19 ? __width(__common_type::period::den) : 6u;
  using precision = duration<typename __common_type::rep, ratio<1, __pow10(fractional_width)>>;

  _CCCL_API constexpr hh_mm_ss() noexcept
      : hh_mm_ss{_Duration::zero()}
  {}

  _CCCL_API constexpr explicit hh_mm_ss(_Duration __dur) noexcept
      : __is_neg_(__dur < _Duration(0))
      , __hours_(::cuda::std::chrono::duration_cast<chrono::hours>(::cuda::std::chrono::abs(__dur)))
      , __month_(::cuda::std::chrono::duration_cast<chrono::minutes>(::cuda::std::chrono::abs(__dur) - hours()))
      , __seconds_(
          ::cuda::std::chrono::duration_cast<chrono::seconds>(::cuda::std::chrono::abs(__dur) - hours() - minutes()))
      , __fraction_(::cuda::std::chrono::duration_cast<precision>(
          ::cuda::std::chrono::abs(__dur) - hours() - minutes() - seconds()))
  {}

  [[nodiscard]] _CCCL_API constexpr bool is_negative() const noexcept
  {
    return __is_neg_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::hours hours() const noexcept
  {
    return __hours_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::minutes minutes() const noexcept
  {
    return __month_;
  }

  [[nodiscard]] _CCCL_API constexpr chrono::seconds seconds() const noexcept
  {
    return __seconds_;
  }

  [[nodiscard]] _CCCL_API constexpr precision subseconds() const noexcept
  {
    return __fraction_;
  }

  [[nodiscard]] _CCCL_API constexpr precision to_duration() const noexcept
  {
    const auto __dur = __hours_ + __month_ + __seconds_ + __fraction_;
    return __is_neg_ ? -__dur : __dur;
  }

  [[nodiscard]] _CCCL_API constexpr explicit operator precision() const noexcept
  {
    return to_duration();
  }

private:
  bool __is_neg_;
  chrono::hours __hours_;
  chrono::minutes __month_;
  chrono::seconds __seconds_;
  precision __fraction_;
};

[[nodiscard]] _CCCL_API constexpr bool is_am(const hours& __hours_) noexcept
{
  return __hours_ >= hours{0} && __hours_ < hours{12};
}

[[nodiscard]] _CCCL_API constexpr bool is_pm(const hours& __hours_) noexcept
{
  return __hours_ >= hours{12} && __hours_ < hours{24};
}

[[nodiscard]] _CCCL_API constexpr hours make12(const hours& __hours_) noexcept
{
  if (__hours_ == hours{0})
  {
    return hours{12};
  }
  else if (__hours_ <= hours{12})
  {
    return __hours_;
  }
  else
  {
    return __hours_ - hours{12};
  }
}

[[nodiscard]] _CCCL_API constexpr hours make24(const hours& __hours_, bool __is_pm) noexcept
{
  if (__is_pm)
  {
    return __hours_ == hours{12} ? __hours_ : __hours_ + hours{12};
  }
  else
  {
    return __hours_ == hours{12} ? hours{0} : __hours_;
  }
}
} // namespace chrono

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CHRONO_HH_MM_SS_H
