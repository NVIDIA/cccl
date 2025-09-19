//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___KERNEL_TRAITS
#define _CUDAX___KERNEL_TRAITS

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{

//! @brief A variable that can be specialized to `true` for types that don't satisfy the `kernel_argument` concept
//!        but should be treated as valid kernel argument types anyway.
//!
//! @warning This violates the standard C++ model and can lead to undefined behaviour and hard-to-resolve bugs.
//!
//! @tparam _Tp The type to proclaim.
template <class _Tp>
inline constexpr bool proclaim_kernel_argument_v = false;

} // namespace cuda::experimental

#endif // _CUDAX___KERNEL_TRAITS
