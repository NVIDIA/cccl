//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_DECAY_H
#define _CUDA_STD___TYPE_TRAITS_DECAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_referenceable.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/std/__cccl/prologue.h>

#if (_CCCL_CHECK_BUILTIN(decay) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX(15))
#  define _CCCL_BUILTIN_DECAY(...) __decay(__VA_ARGS__)
#endif // (_CCCL_CHECK_BUILTIN(decay) && !_CCCL_BUILTIN_CONFLICTS_WITH_LIBSTDCXX( 15))

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVRTC) // NVCC has issues with function pointers see nvbug6665129
#  undef _CCCL_BUILTIN_DECAY
#endif // _CCCL_CUDA_COMPILER(NVCC)

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if defined(_CCCL_BUILTIN_DECAY) && !defined(_LIBCUDACXX_USE_DECAY_FALLBACK)
template <class _Tp>
struct decay
{
  using type _CCCL_NODEBUG = _CCCL_BUILTIN_DECAY(_Tp);
};

#  if _CCCL_COMPILER(GCC) // GCC does not accept the builtin in template signatures
template <class _Tp>
using decay_t _CCCL_NODEBUG = typename decay<_Tp>::type;
#  else // ^^^ _CCCL_COMPILER(GCC) ^^^ / vvv !_CCCL_COMPILER(GCC) vvv
template <class _Tp>
using decay_t _CCCL_NODEBUG = _CCCL_BUILTIN_DECAY(_Tp);
#  endif // !_CCCL_COMPILER(GCC)

#else // ^^^ _CCCL_BUILTIN_DECAY ^^^ / vvv !_CCCL_BUILTIN_DECAY vvv

template <class _Up, bool>
struct __decay_impl
{
  using type _CCCL_NODEBUG = remove_cv_t<_Up>;
};

template <class _Up>
struct __decay_impl<_Up, true>
{
public:
  using type _CCCL_NODEBUG =
    conditional_t<is_array_v<_Up>,
                  remove_extent_t<_Up>*,
                  conditional_t<is_function_v<_Up>, add_pointer_t<_Up>, remove_cv_t<_Up>>>;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT decay
{
private:
  using _Up _CCCL_NODEBUG = remove_reference_t<_Tp>;

public:
  using type _CCCL_NODEBUG = typename __decay_impl<_Up, __cccl_is_referenceable<_Up>::value>::type;
};

template <class _Tp>
using decay_t _CCCL_NODEBUG = typename decay<_Tp>::type;

#endif // !_CCCL_BUILTIN_DECAY

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_DECAY_H
