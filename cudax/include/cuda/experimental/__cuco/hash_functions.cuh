//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_HASH_FUNCTIONS_CUH
#define _CUDAX__CUCO_HASH_FUNCTIONS_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/functional.h>

#include <cuda/experimental/__cuco/detail/hash_functions/xxhash.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{

//! @brief A 32-bit `XXH32` hash function to hash the given argument on host and device.
//!
//! @tparam Key The type of the values to hash
template <typename Key>
using xxhash_32 = __detail::XXHash_32<Key>;

} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HASH_FUNCTIONS_CUH
