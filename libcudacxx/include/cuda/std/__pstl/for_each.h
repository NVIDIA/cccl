//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___PSTL_FOR_EACH_H
#define _LIBCUDACXX___PSTL_FOR_EACH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/for_each_n.h>
#include <cuda/std/__execution/policy.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__pstl/dispatch.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <::cuda::std::execution::__execution_policy _Policy, class _ForwardIterator, class _Size, class _Function>
_CCCL_API void for_each_n(
  const ::cuda::std::execution::__policy<_Policy>& __pol, _ForwardIterator __first, _Size __orig_n, _Function __func)
{
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__for_each_n, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    __dispatch(__pol, ::cuda::std::move(__first), __orig_n, ::cuda::std::move(__func));
  }
  else
  {
    ::cuda::std::for_each_n(::cuda::std::move(__first), __orig_n, ::cuda::std::move(__func));
  }
}

template <::cuda::std::execution::__execution_policy _Policy, class _ForwardIterator, class _Function>
_CCCL_API void for_each(const ::cuda::std::execution::__policy<_Policy>& __pol,
                        _ForwardIterator __first,
                        _ForwardIterator __last,
                        _Function __func)
{
  [[maybe_unused]] auto __dispatch =
    ::cuda::std::execution::__pstl_select_dispatch<::cuda::std::execution::__pstl_algorithm::__for_each_n, _Policy>();
  if constexpr (::cuda::std::execution::__pstl_can_dispatch<decltype(__dispatch)>)
  {
    __dispatch(__pol, ::cuda::std::move(__first), ::cuda::std::distance(__first, __last), ::cuda::std::move(__func));
  }
  else
  {
    ::cuda::std::for_each(::cuda::std::move(__first), ::cuda::std::move(__last), ::cuda::std::move(__func));
  }
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___PSTL_FOR_EACH_H
