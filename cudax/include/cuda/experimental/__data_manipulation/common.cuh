//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DATA_MANIPULATION_COMMON
#define __CUDAX_DATA_MANIPULATION_COMMON

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/span>

#include <cuda/experimental/__launch/launch_transform.cuh>

namespace cuda::experimental
{

template <typename _Ty>
_CCCL_INLINE_VAR constexpr bool __is_span = false;

template <typename _TyInSpan, ::std::size_t _Size>
_CCCL_INLINE_VAR constexpr bool __is_span<::cuda::std::span<_TyInSpan, _Size>> = true;

// TODO add more cases like mdspan
template <typename _Ty>
_CCCL_INLINE_VAR constexpr bool __transforms_to_copy_fill_arg = __is_span<as_kernel_arg_t<_Ty>>;

} // namespace cuda::experimental
#endif // __CUDAX_DATA_MANIPULATION_COMMON
