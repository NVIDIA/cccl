//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_IS_ALLOCATOR_H
#define _LIBCUDACXX___TYPE_IS_ALLOCATOR_H

#ifndef __cuda_std__
#  include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstddef>
#include <cuda/std/detail/libcxx/include/__type_traits/integral_constant.h>
#include <cuda/std/detail/libcxx/include/__type_traits/void_t.h>
#include <cuda/std/detail/libcxx/include/__utility/declval.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <typename _Alloc, typename = void, typename = void>
struct __is_allocator : false_type
{};

template <typename _Alloc>
struct __is_allocator<_Alloc,
                      __void_t<typename _Alloc::value_type>,
                      __void_t<decltype(_CUDA_VSTD::declval<_Alloc&>().allocate(size_t(0)))>> : true_type
{};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_IS_ALLOCATOR_H
