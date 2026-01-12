//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_NAMESPACES_H
#define _CUDA_STD___INTERNAL_NAMESPACES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__internal/version.h>

// During the header testing, we want to check if the code is wrapped by the prologue/epilogue
#if defined(_CCCL_HEADER_TEST)
#  define _CCCL_PROLOGUE_INCLUDE_CHECK() \
    static_assert(_CCCL_PROLOGUE_INCLUDED(), "missing #include <cuda/std/__cccl/prologue.h>");
#else // ^^^ defined(_CCCL_HEADER_TEST) ^^^ / vvv !defined(_CCCL_HEADER_TEST) vvv
#  define _CCCL_PROLOGUE_INCLUDE_CHECK()
#endif // ^^^ !defined(_CCCL_HEADER_TEST) ^^^

#ifndef _LIBCUDACXX_ABI_NAMESPACE
#  define _LIBCUDACXX_ABI_NAMESPACE _CCCL_PP_CAT(__, _LIBCUDACXX_CUDA_ABI_VERSION)
#endif // _LIBCUDACXX_ABI_NAMESPACE

#define _CCCL_BEGIN_NAMESPACE_NOVERSION(_NS)   \
  _CCCL_PROLOGUE_INCLUDE_CHECK() namespace _NS \
  {
#define _CCCL_END_NAMESPACE_NOVERSION(_NS) \
  }                                        \
  _CCCL_PROLOGUE_INCLUDE_CHECK()
#define _CCCL_BEGIN_NAMESPACE(_NS)                                                \
  _CCCL_BEGIN_NAMESPACE_NOVERSION(_NS) inline namespace _LIBCUDACXX_ABI_NAMESPACE \
  {
#define _CCCL_END_NAMESPACE(_NS) \
  }                              \
  _CCCL_END_NAMESPACE_NOVERSION(_NS)

// Standard namespaces with or without versioning
#define _CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION _CCCL_BEGIN_NAMESPACE_NOVERSION(cuda::std)
#define _CCCL_END_NAMESPACE_CUDA_STD_NOVERSION   _CCCL_END_NAMESPACE_NOVERSION(cuda::std)
#define _CCCL_BEGIN_NAMESPACE_CUDA_STD           _CCCL_BEGIN_NAMESPACE(cuda::std)
#define _CCCL_END_NAMESPACE_CUDA_STD             _CCCL_END_NAMESPACE(cuda::std)

// cuda specific namespaces
#define _CCCL_BEGIN_NAMESPACE_CUDA                     _CCCL_BEGIN_NAMESPACE(cuda)
#define _CCCL_END_NAMESPACE_CUDA                       _CCCL_END_NAMESPACE(cuda)
#define _CCCL_BEGIN_NAMESPACE_CUDA_MR                  _CCCL_BEGIN_NAMESPACE(cuda::mr)
#define _CCCL_END_NAMESPACE_CUDA_MR                    _CCCL_END_NAMESPACE(cuda::mr)
#define _CCCL_BEGIN_NAMESPACE_CUDA_DEVICE              _CCCL_BEGIN_NAMESPACE(cuda::device)
#define _CCCL_END_NAMESPACE_CUDA_DEVICE                _CCCL_END_NAMESPACE(cuda::device)
#define _CCCL_BEGIN_NAMESPACE_CUDA_PTX                 _CCCL_BEGIN_NAMESPACE(cuda::ptx)
#define _CCCL_END_NAMESPACE_CUDA_PTX                   _CCCL_END_NAMESPACE(cuda::ptx)
#define _CCCL_BEGIN_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL _CCCL_BEGIN_NAMESPACE(cuda::device::experimental)
#define _CCCL_END_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL   _CCCL_END_NAMESPACE(cuda::device::experimental)
#define _CCCL_BEGIN_NAMESPACE_CUDA_DRIVER              _CCCL_BEGIN_NAMESPACE(cuda::__driver)
#define _CCCL_END_NAMESPACE_CUDA_DRIVER                _CCCL_END_NAMESPACE(cuda::__driver)

// Namespaces related to <ranges>
#define _CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES _CCCL_BEGIN_NAMESPACE(cuda::std::ranges)
#define _CCCL_END_NAMESPACE_CUDA_STD_RANGES   _CCCL_END_NAMESPACE(cuda::std::ranges)
#define _CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS  _CCCL_BEGIN_NAMESPACE(cuda::std::ranges::views)
#define _CCCL_END_NAMESPACE_CUDA_STD_VIEWS    _CCCL_END_NAMESPACE(cuda::std::ranges::views)

#define _CCCL_BEGIN_NAMESPACE_CPO(_CPO) \
  namespace _CPO                        \
  {
#define _CCCL_END_NAMESPACE_CPO }

// Namespaces related to chrono / filesystem
#define _CCCL_BEGIN_NAMESPACE_FILESYSTEM     \
  _CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION   \
  inline namespace __fs                      \
  {                                          \
  namespace filesystem                       \
  {                                          \
  inline namespace _LIBCUDACXX_ABI_NAMESPACE \
  {
#define _CCCL_END_NAMESPACE_FILESYSTEM \
  }                                    \
  }                                    \
  }                                    \
  _CCCL_END_NAMESPACE_CUDA_STD_NOVERSION

// Shorthands for different qualifiers
// Namespaces related to execution
#define _CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION _CCCL_BEGIN_NAMESPACE(cuda::std::execution)
#define _CCCL_END_NAMESPACE_CUDA_STD_EXECUTION   _CCCL_END_NAMESPACE(cuda::std::execution)

#define _CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION _CCCL_BEGIN_NAMESPACE(cuda::execution)
#define _CCCL_END_NAMESPACE_CUDA_EXECUTION   _CCCL_END_NAMESPACE(cuda::execution)

// Namespace to avoid name collisions with CPOs on clang-16 (see https://godbolt.org/z/9TadonrdM for example)
#if _CCCL_COMPILER(CLANG, <=, 16)
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE \
    namespace __hidden                              \
    {
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS) \
    }                                                     \
    using __hidden::_CLASS;
#else // ^^^ _CCCL_COMPILER(CLANG, <=, 16) ^^^ / vvv _CCCL_COMPILER(CLANG, >, 16) vvv
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS)
#endif // !_CCCL_COMPILER(CLANG, >, 16)

#if defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)
#  define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
#  define _CCCL_END_NAMESPACE_ARCH_DEPENDENT
#else // not defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)
#  if _CCCL_CUDA_COMPILER(NVHPC)
#    define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT                                                     \
      inline namespace _CCCL_PP_CAT(_CCCL_PP_SPLICE_WITH(_, _SM, NV_TARGET_SM_INTEGER_LIST), _NVHPC) \
      {
#    define _CCCL_END_NAMESPACE_ARCH_DEPENDENT }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVHPC) ^^^ / vvv !_CCCL_CUDA_COMPILER(NVHPC) vvv
#    define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT                        \
      inline namespace _CCCL_PP_SPLICE_WITH(_, _SM, __CUDA_ARCH_LIST__) \
      {
#    define _CCCL_END_NAMESPACE_ARCH_DEPENDENT }
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVHPC) ^^^
#endif // not defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)

// Host standard library namespaces
#if _CCCL_HOST_STD_LIB(LIBSTDCXX)
// We don't appy attributes on forward declarations, so we omit the _GLIBCXX_VISIBILITY(default)
#  if _GLIBCXX_INLINE_VERSION
#    define _CCCL_BEGIN_NAMESPACE_STD              \
      _CCCL_PROLOGUE_INCLUDE_CHECK() namespace std \
      {                                            \
        inline _GLIBCXX_BEGIN_NAMESPACE_VERSION
#    define _CCCL_END_NAMESPACE_STD  \
      _GLIBCXX_END_NAMESPACE_VERSION \
      }                              \
      _CCCL_PROLOGUE_INCLUDE_CHECK()
#  else // ^^^ _GLIBCXX_INLINE_VERSION ^^^ / vvv !_GLIBCXX_INLINE_VERSION vvv
#    define _CCCL_BEGIN_NAMESPACE_STD              \
      _CCCL_PROLOGUE_INCLUDE_CHECK() namespace std \
      {
#    define _CCCL_END_NAMESPACE_STD \
      }                             \
      _CCCL_PROLOGUE_INCLUDE_CHECK()
#  endif // ^^^ !_GLIBCXX_INLINE_VERSION ^^^
#elif _CCCL_HOST_STD_LIB(LIBCXX)
#  define _CCCL_BEGIN_NAMESPACE_STD _CCCL_PROLOGUE_INCLUDE_CHECK() _LIBCPP_BEGIN_NAMESPACE_STD
#  define _CCCL_END_NAMESPACE_STD   _LIBCPP_END_NAMESPACE_STD _CCCL_PROLOGUE_INCLUDE_CHECK()
#elif _CCCL_HOST_STD_LIB(STL)
#  define _CCCL_BEGIN_NAMESPACE_STD _CCCL_PROLOGUE_INCLUDE_CHECK() _STD_BEGIN
#  define _CCCL_END_NAMESPACE_STD   _STD_END _CCCL_PROLOGUE_INCLUDE_CHECK()
#else
#  define _CCCL_BEGIN_NAMESPACE_STD              \
    _CCCL_PROLOGUE_INCLUDE_CHECK() namespace std \
    {
#  define _CCCL_END_NAMESPACE_STD \
    }                             \
    _CCCL_PROLOGUE_INCLUDE_CHECK()
#endif

#endif // _CUDA_STD___INTERNAL_NAMESPACES_H
