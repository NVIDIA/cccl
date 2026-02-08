//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H
#define _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__floating_point/traits.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_nothrow_convertible.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp, typename _Up, typename _Common = ::cuda::std::common_type_t<_Tp, _Up>, typename _Enable = void>
constexpr bool __is_maximum_minimum_noexcept_v =
  noexcept(::cuda::std::declval<_Tp>() < ::cuda::std::declval<_Up>())
  && ::cuda::std::is_nothrow_convertible_v<_Tp, _Common> && ::cuda::std::is_nothrow_convertible_v<_Up, _Common>;

// Extended floating point types, such as __half and __nv bfloat16 cannot be compared with operator<. We need to
// handle them separately with SFINAE.
template <typename _Tp, typename _Up, typename _Common>
constexpr bool __is_maximum_minimum_noexcept_v<
  _Tp,
  _Up,
  _Common,
  ::cuda::std::enable_if_t<::cuda::std::__is_ext_nv_fp_v<_Tp> || ::cuda::std::__is_ext_nv_fp_v<_Up>>> = false;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MINIMUM_MAXIMUM_COMMON_H
