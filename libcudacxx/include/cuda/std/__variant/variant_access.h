//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VARIANT_ACCESS_H
#define _CUDA_STD___VARIANT_VARIANT_ACCESS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/in_place.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __variant_detail::__access
{
struct __union
{
  template <class _Vp>
  [[nodiscard]] _CCCL_API static constexpr auto&& __get_alt(_Vp&& __v, in_place_index_t<0>) noexcept
  {
    return ::cuda::std::forward<_Vp>(__v).__head_;
  }

  template <class _Vp, size_t _Ip>
  [[nodiscard]] _CCCL_API static constexpr auto&& __get_alt(_Vp&& __v, in_place_index_t<_Ip>) noexcept
  {
    return __get_alt(::cuda::std::forward<_Vp>(__v).__tail_, in_place_index<_Ip - 1>);
  }
};

struct __base
{
  template <size_t _Ip, class _Vp>
  [[nodiscard]] _CCCL_API static constexpr auto&& __get_alt(_Vp&& __v) noexcept
  {
    return __union::__get_alt(::cuda::std::forward<_Vp>(__v).__data_, in_place_index<_Ip>);
  }
};

struct __variant
{
  template <size_t _Ip, class _Vp>
  [[nodiscard]] _CCCL_API static constexpr auto&& __get_alt(_Vp&& __v) noexcept
  {
    return __base::__get_alt<_Ip>(::cuda::std::forward<_Vp>(__v).__impl_);
  }
};
} // namespace __variant_detail::__access

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VARIANT_ACCESS_H
