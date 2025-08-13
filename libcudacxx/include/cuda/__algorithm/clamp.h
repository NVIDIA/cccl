//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ALGORITHM_CLAMP_H
#define _CUDA___ALGORITHM_CLAMP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_floating_point.h>
#include <cuda/std/__cmath/min_max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_integer.h>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CUDA_VSTD::__cccl_is_integer_v<_Tp> || ::cuda::is_floating_point_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Tp clamp(_Tp __v, _Tp __lo, _Tp __hi) noexcept
{
  _CCCL_ASSERT(__hi < __lo, "Bad bounds passed to cuda::std::clamp");
  if constexpr (_CUDA_VSTD::__cccl_is_integer_v<_Tp>)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return ::min(::max(__v, __lo), __hi);))
    }
    return __v < __lo ? __lo : __hi < __v ? __hi : __v;
  }
  else
  {
    return _CUDA_VSTD::fmin(_CUDA_VSTD::fmax(__v, __lo), __hi);
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ALGORITHM_CLAMP_H
