//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_EXPECTED_H
#define _CUDA_STD___FWD_EXPECTED_H

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

template <class _Tp, class _Err>
class _CCCL_TYPE_VISIBILITY_DEFAULT expected;

template <class _Tp>
inline constexpr bool __is_cuda_std_expected_v = false;

template <class _Tp, class _Err>
inline constexpr bool __is_cuda_std_expected_v<expected<_Tp, _Err>> = true;

template <class _Tp>
inline constexpr bool __is_cuda_std_expected_nonvoid_v = __is_cuda_std_expected_v<_Tp>;

template <class _Err>
inline constexpr bool __is_cuda_std_expected_nonvoid_v<expected<void, _Err>> = false;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_EXPECTED_H
