//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___INTERNAL_NAMESPACES_H
#define _LIBCUDACXX___INTERNAL_NAMESPACES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#define _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y) _LIBCUDACXX_X##_LIBCUDACXX_Y
#define _LIBCUDACXX_CONCAT(_LIBCUDACXX_X, _LIBCUDACXX_Y)  _LIBCUDACXX_CONCAT1(_LIBCUDACXX_X, _LIBCUDACXX_Y)

#ifndef _LIBCUDACXX_ABI_NAMESPACE
#  define _LIBCUDACXX_ABI_NAMESPACE _LIBCUDACXX_CONCAT(__, _LIBCUDACXX_CUDA_ABI_VERSION)
#endif // _LIBCUDACXX_ABI_NAMESPACE

// clang-format off

// Standard namespaces with or without versioning
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD_NOVERSION namespace cuda { namespace std {
#  define _LIBCUDACXX_END_NAMESPACE_STD_NOVERSION } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_STD namespace cuda { namespace std { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_STD } } }

// cuda specific namespaces
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA namespace cuda { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR namespace cuda { namespace mr { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_MR } } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE namespace cuda { namespace device { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE } } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_PTX namespace cuda { namespace ptx { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_PTX } } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL namespace cuda { namespace device { namespace experimental { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE_EXPERIMENTAL } } } }

// Namespaces related to <ranges>
#  define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES namespace cuda { namespace std { namespace ranges { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_RANGES } } } }
#  define _LIBCUDACXX_BEGIN_NAMESPACE_VIEWS namespace cuda { namespace std { namespace ranges { namespace views { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_VIEWS } } } } }

#  if _CCCL_STD_VER >= 2020
#    define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI inline namespace __cxx20 {
#  else
#    define _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI inline namespace __cxx17 {
#  endif
#  define _LIBCUDACXX_END_NAMESPACE_RANGES_ABI }

#  define _LIBCUDACXX_BEGIN_NAMESPACE_CPO(_CPO) namespace _CPO { _LIBCUDACXX_BEGIN_NAMESPACE_RANGES_ABI
#  define _LIBCUDACXX_END_NAMESPACE_CPO } _LIBCUDACXX_END_NAMESPACE_RANGES_ABI

// Namespaces related to chrono / filesystem
#  if _CCCL_STD_VER >= 2017
#    define _LIBCUDACXX_BEGIN_NAMESPACE_FILESYSTEM namespace cuda { namespace std { inline namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  else // ^^^ C++17 ^^^ / vvv C++14 vvv
#    define _LIBCUDACXX_BEGIN_NAMESPACE_FILESYSTEM namespace cuda { namespace std { namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  endif // _CCCL_STD_VER <= 2014
#  define _LIBCUDACXX_END_NAMESPACE_FILESYSTEM } } } } }

// Shorthands for different qualifiers
#  define _CUDA_VSTD_NOVERSION ::cuda::std
#  define _CUDA_VSTD           ::cuda::std::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VRANGES        ::cuda::std::ranges::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VIEWS          ::cuda::std::ranges::views::_LIBCUDACXX_CUDA_ABI_NAMESPACE
#  define _CUDA_VMR            ::cuda::mr::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VPTX           ::cuda::ptx::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VSTD_FS        ::cuda::std::__fs::filesystem::_LIBCUDACXX_ABI_NAMESPACE

// clang-format on

#endif // _LIBCUDACXX___INTERNAL_NAMESPACES_H
