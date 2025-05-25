//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_LANE_MASK_H
#define _CUDA___WARP_LANE_MASK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/__ptx/instructions/get_sreg.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/is_integer.h>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

struct lane_mask
{
  using __value_type = uint32_t;

  __value_type value;

  _CCCL_HIDE_FROM_ABI constexpr lane_mask() noexcept = default;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_integer_v<_Tp> _CCCL_AND(sizeof(_Tp) == sizeof(__value_type)))
  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI explicit constexpr lane_mask(_Tp __v) noexcept
      : value{static_cast<__value_type>(__v)}
  {}

  _CCCL_HIDE_FROM_ABI constexpr lane_mask(const lane_mask&) noexcept = default;

  _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator=(const lane_mask&) noexcept = default;

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static constexpr lane_mask empty() noexcept
  {
    return lane_mask{};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static constexpr lane_mask all() noexcept
  {
    return lane_mask{_CUDA_VSTD::numeric_limits<__value_type>::max()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_active() noexcept
  {
    return lane_mask{::__activemask()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask this_lane() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_eq()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_less() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_lt()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_less() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_lt()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_less_equal() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_le()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_greater() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_gt()};
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI static lane_mask all_greater_equal() noexcept
  {
    return lane_mask{_CUDA_VPTX::get_sreg_lanemask_ge()};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator&(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.value & __rhs.value};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator&=(lane_mask __v) noexcept
  {
    return *this = *this & __v;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator|(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.value | __rhs.value};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator|=(lane_mask __v) noexcept
  {
    return *this = *this | __v;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator^(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return lane_mask{__lhs.value ^ __rhs.value};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator^=(lane_mask __v) noexcept
  {
    return *this = *this ^ __v;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator<<(lane_mask __mask, int __shift) noexcept
  {
    return lane_mask{__mask.value << __shift};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator<<=(int __shift) noexcept
  {
    return *this = *this << __shift;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask
  operator>>(lane_mask __mask, int __shift) noexcept
  {
    return lane_mask{__mask.value >> __shift};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr lane_mask& operator>>=(int __shift) noexcept
  {
    return *this = *this >> __shift;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr lane_mask operator~(lane_mask __mask) noexcept
  {
    return lane_mask{~__mask.value};
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator==(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.value == __rhs.value;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator!=(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator<(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.value < __rhs.value;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator<=(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.value <= __rhs.value;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator>(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.value > __rhs.value;
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_HIDE_FROM_ABI friend constexpr bool
  operator>=(lane_mask __lhs, lane_mask __rhs) noexcept
  {
    return __lhs.value >= __rhs.value;
  }

  _CCCL_DEVICE _CCCL_HIDE_FROM_ABI constexpr operator __value_type() const noexcept
  {
    return value;
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___WARP_LANE_MASK_H
