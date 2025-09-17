//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_CONCEPTS
#define _CUDAX___KERNEL_CONCEPTS

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>

namespace cuda::experimental
{

//! @brief Concept that checks if a type is a valid kernel argument type.
//!
//! @details A valid kernel argument type is one that is trivially copyable and trivially destructible. For more details
//!          see https://docs.nvidia.com/cuda/cuda-c-programming-guide/#global-function-argument-processing.
//!
/// @tparam _Tp The type to check.
template <class _Tp>
_CCCL_CONCEPT kernel_argument =
  ::cuda::std::is_trivially_copyable_v<_Tp> && ::cuda::std::is_trivially_destructible_v<_Tp>;

} // namespace cuda::experimental

#endif // _CUDAX___KERNEL_CONCEPTS
