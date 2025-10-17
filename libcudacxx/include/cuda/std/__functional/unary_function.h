//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_UNARY_FUNCTION_H
#define _CUDA_STD___FUNCTIONAL_UNARY_FUNCTION_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

template <class _Arg, class _Result>
struct _CCCL_TYPE_VISIBILITY_DEFAULT CCCL_DEPRECATED unary_function
{
  using argument_type = _Arg;
  using result_type   = _Result;
};

#endif // _LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION

template <class _Arg, class _Result>
struct __unary_function_keep_layout_base
{
#if _CCCL_STD_VER <= 2017 || defined(_LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS)
  using argument_type CCCL_DEPRECATED = _Arg;
  using result_type CCCL_DEPRECATED   = _Result;
#endif
};

#if defined(_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)

_CCCL_SUPPRESS_DEPRECATED_PUSH
template <class _Arg, class _Result>
using __unary_function = unary_function<_Arg, _Result>;
_CCCL_SUPPRESS_DEPRECATED_POP

#else
template <class _Arg, class _Result>
using __unary_function = __unary_function_keep_layout_base<_Arg, _Result>;
#endif // !_LIBCUDACXX_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_UNARY_FUNCTION_H
