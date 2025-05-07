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

#  define _LIBCUDACXX_BEGIN_NAMESPACE_CPO(_CPO) namespace _CPO {
#  define _LIBCUDACXX_END_NAMESPACE_CPO }

// Namespaces related to chrono / filesystem
#  define _LIBCUDACXX_BEGIN_NAMESPACE_FILESYSTEM namespace cuda { namespace std { inline namespace __fs { namespace filesystem { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_FILESYSTEM } } } } }

// Shorthands for different qualifiers
  // Namespaces related to execution
#  define _LIBCUDACXX_BEGIN_NAMESPACE_EXECUTION namespace cuda { namespace std { namespace execution { inline namespace _LIBCUDACXX_ABI_NAMESPACE {
#  define _LIBCUDACXX_END_NAMESPACE_EXECUTION } } } }

// Namespace to avoid name collisions with CPOs on clang-16 (see https://godbolt.org/z/9TadonrdM for example)
#if _CCCL_COMPILER(CLANG, ==, 16)
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE namespace __hidden {
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS) } using __hidden::_CLASS;
#else // ^^^ _CCCL_COMPILER(CLANG, ==, 16) ^^^ / vvv !_CCCL_COMPILER(CLANG, ==, 16) vvv
#  define _LIBCUDACXX_BEGIN_HIDDEN_FRIEND_NAMESPACE
#  define _LIBCUDACXX_END_HIDDEN_FRIEND_NAMESPACE(_CLASS)
#endif // !_CCCL_COMPILER(CLANG, ==, 16)

  // Shorthands for different qualifiers
#  define _CUDA_VSTD_NOVERSION ::cuda::std
#  define _CUDA_VSTD           ::cuda::std::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VRANGES        ::cuda::std::ranges::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VIEWS          ::cuda::std::ranges::views::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VMR            ::cuda::mr::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VPTX           ::cuda::ptx::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_VSTD_FS        ::cuda::std::__fs::filesystem::_LIBCUDACXX_ABI_NAMESPACE
#  define _CUDA_STD_EXEC       ::cuda::std::execution::_LIBCUDACXX_ABI_NAMESPACE

// clang-format on

#endif // _LIBCUDACXX___INTERNAL_NAMESPACES_H
