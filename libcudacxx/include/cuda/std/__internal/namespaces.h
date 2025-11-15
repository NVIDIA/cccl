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
#  define _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() \
    static_assert(_CCCL_PROLOGUE_INCLUDED(), "missing #include <cuda/std/__cccl/prologue.h>");
#else // ^^^ defined(_CCCL_HEADER_TEST) ^^^ / vvv !defined(_CCCL_HEADER_TEST) vvv
#  define _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#endif // ^^^ !defined(_CCCL_HEADER_TEST) ^^^

#define _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y) _LIBCUDACXX_X##_LIBCUDACXX_Y
#define _LIBCUDACXX_CONCAT(_LIBCUDACXX_X, _LIBCUDACXX_Y)  _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y)

#ifndef _LIBCUDACXX_ABI_NAMESPACE
#  define _LIBCUDACXX_ABI_NAMESPACE _LIBCUDACXX_CONCAT(__, _LIBCUDACXX_CUDA_ABI_VERSION)
#endif // _LIBCUDACXX_ABI_NAMESPACE

// clang-format off

// Standard namespaces with or without versioning
#  define _CCCL_BEGIN_NAMESPACE_CUDA_STD_NOVERSION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std {
#  define _CCCL_END_NAMESPACE_CUDA_STD_NOVERSION } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_STD } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

// cuda specific namespaces
#  define _CCCL_BEGIN_NAMESPACE_CUDA _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_MR _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::mr { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_MR } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_DEVICE _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::device { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_DEVICE } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_PTX _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::ptx { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_PTX } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::device::experimental { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_DRIVER _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::__driver { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_DRIVER } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

// Namespaces related to <ranges>
#  define _CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::ranges { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_STD_RANGES } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  define _CCCL_BEGIN_NAMESPACE_CUDA_STD_VIEWS _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::ranges::views { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_STD_VIEWS } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

#  define _CCCL_BEGIN_NAMESPACE_CPO(_CPO) namespace _CPO {
#  define _CCCL_END_NAMESPACE_CPO }

// Namespaces related to chrono / filesystem
#  define _CCCL_BEGIN_NAMESPACE_FILESYSTEM _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std { inline namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_FILESYSTEM } } } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

// Shorthands for different qualifiers
// Namespaces related to execution
#  define _CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda::std::execution { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_STD_EXECUTION } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

#  define _CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace cuda { namespace execution { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _CCCL_END_NAMESPACE_CUDA_EXECUTION } } } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()

// Namespace to avoid name collisions with CPOs on clang-16 (see https://godbolt.org/z/9TadonrdM for example)
#if _CCCL_COMPILER(CLANG, <=, 16)
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE namespace __hidden {
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS) } using __hidden::_CLASS;
#else // ^^^ _CCCL_COMPILER(CLANG, <=, 16) ^^^ / vvv _CCCL_COMPILER(CLANG, >, 16) vvv
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS)
#endif // !_CCCL_COMPILER(CLANG, >, 16)

#if defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)
#  define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT
#  define _CCCL_END_NAMESPACE_ARCH_DEPENDENT
#else // not defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)
#  if defined(_NVHPC_CUDA)
#    define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT                                                    \
      inline namespace _CCCL_PP_CAT(_CCCL_PP_SPLICE_WITH(_, _SM, NV_TARGET_SM_INTEGER_LIST), _NVHPC) \
      {
#    define _CCCL_END_NAMESPACE_ARCH_DEPENDENT }
#  else // not defined(_NVHPC_CUDA)
#    define _CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT                                                    \
      inline namespace _CCCL_PP_SPLICE_WITH(_, _SM, __CUDA_ARCH_LIST__)                \
      {
#    define _CCCL_END_NAMESPACE_ARCH_DEPENDENT }
#  endif // not defined(_NVHPC_CUDA)
#endif // not defined(CCCL_DISABLE_ARCH_DEPENDENT_NAMESPACE)

// Host standard library namespaces
#if _CCCL_HOST_STD_LIB(LIBSTDCXX)
  // We don't appy attributes on forward declarations, so we omit the _GLIBCXX_VISIBILITY(default)
#  if _GLIBCXX_INLINE_VERSION
#    define _CCCL_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace std { inline _GLIBCXX_BEGIN_NAMESPACE_VERSION
#    define _CCCL_END_NAMESPACE_STD   _GLIBCXX_END_NAMESPACE_VERSION } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  else // ^^^ _GLIBCXX_INLINE_VERSION ^^^ / vvv !_GLIBCXX_INLINE_VERSION vvv
#    define _CCCL_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace std {
#    define _CCCL_END_NAMESPACE_STD   } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#  endif // ^^^ !_GLIBCXX_INLINE_VERSION ^^^
#elif _CCCL_HOST_STD_LIB(LIBCXX)
#  define _CCCL_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() _LIBCPP_BEGIN_NAMESPACE_STD
#  define _CCCL_END_NAMESPACE_STD   _LIBCPP_END_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#elif _CCCL_HOST_STD_LIB(STL)
#  define _CCCL_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() _STD_BEGIN
#  define _CCCL_END_NAMESPACE_STD   _STD_END _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#else
#  define _CCCL_BEGIN_NAMESPACE_STD _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK() namespace std {
#  define _CCCL_END_NAMESPACE_STD   } _LIBCUDACXX_PROLOGUE_INCLUDE_CHECK()
#endif

// clang-format on

#endif // _CUDA_STD___INTERNAL_NAMESPACES_H
