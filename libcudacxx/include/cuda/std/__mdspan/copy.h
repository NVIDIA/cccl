//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MDSPAN_COPY_H
#define _CUDA_STD___MDSPAN_COPY_H

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
#include <cuda/std/__type_traits/is_constructible.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Src, class _Dst>
struct __mdspan_copy_fn
{
  template <class... _Is>
  _CCCL_API constexpr void operator()(_Is... __is) const
  {
    __dst_(__is...) = __src_(__is...);
  }

  const _Src& __src_;
  const _Dst& __dst_;
};

_CCCL_TEMPLATE(class _Src, class _Dst)
_CCCL_REQUIRES(__is_cuda_std_mdspan_v<_Src> _CCCL_AND __is_cuda_std_mdspan_v<_Dst> _CCCL_AND
                 is_assignable_v<typename _Dst::reference, typename _Src::reference> _CCCL_AND
                   is_constructible_v<typename _Src::extents_type, typename _Dst::extents_type>)
_CCCL_API constexpr void copy(const _Src& __src, const _Dst& __dst)
{
  _CCCL_ASSERT(__src.extents() == __dst.extents(), "__src and __dst extents must match");
  _CCCL_ASSERT(__dst.is_unique(), "__dst's mapping must be unique");

  ::cuda::std::__for_each_in_extents(__mdspan_copy_fn<_Src, _Dst>{__src, __dst}, __src.extents());
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MDSPAN_COPY_H
