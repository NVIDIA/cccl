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

#include <cuda/experimental/__cuco/detail/hash_functions/xxhash.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{

enum class HashStrategy
{
  XXHash_32
};

//! @brief A hash function class specialized for different hash strategies.
//!
//! @tparam _Key The type of the values to hash
//! @tparam _S The hash strategy to use, defaults to `HashStrategy::XXHash_32`
template <typename _Key, HashStrategy _S = HashStrategy::XXHash_32>
class Hash;

template <typename _Key>
class Hash<_Key, HashStrategy::XXHash_32> : private __detail::_XXHash_32<_Key>
{
public:
  using __detail::_XXHash_32<_Key>::_XXHash_32;
  using __detail::_XXHash_32<_Key>::operator();
};

} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HASH_FUNCTIONS_CUH
