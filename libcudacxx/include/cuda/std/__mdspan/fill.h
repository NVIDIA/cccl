//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MDSPAN_FILL_H
#define _CUDA_STD___MDSPAN_FILL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/for_each_in_extents.h>
#include <cuda/std/__type_traits/is_assignable.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Mdspan, class _Tp>
struct __mdspan_fill_fn
{
  template <class... _Is>
  _CCCL_API constexpr void operator()(_Is... __is) const
  {
    __mdspan_(__is...) = __value_;
  }

  const _Mdspan& __mdspan_;
  const _Tp& __value_;
};

_CCCL_TEMPLATE(class _Dst, class _Tp = typename _Dst::value_type)
_CCCL_REQUIRES(__is_cuda_std_mdspan_v<_Dst> _CCCL_AND is_assignable_v<typename _Dst::reference, const _Tp&>)
_CCCL_API constexpr void fill(const _Dst& __dst, const _Tp& __v)
{
  ::cuda::std::__for_each_in_extents(__mdspan_fill_fn<_Dst, _Tp>{__dst, __v}, __dst.extents());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MDSPAN_FILL_H
