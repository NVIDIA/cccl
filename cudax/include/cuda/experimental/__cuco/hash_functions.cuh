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

#include <cuda/experimental/__cuco/detail/hash_functions/murmurhash3.cuh>
#include <cuda/experimental/__cuco/detail/hash_functions/xxhash.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco
{
enum class hash_algorithm
{
  xxhash_32,
  xxhash_64,
  murmurhash3_32
#if _CCCL_HAS_INT128()
  ,
  murmurhash3_x86_128,
  murmurhash3_x64_128
#endif // _CCCL_HAS_INT128()
};

//! @brief A hash function class specialized for different hash algorithms.
//!
//! @tparam _Key The type of the values to hash
//! @tparam _S The hash strategy to use, defaults to `hash_algorithm::xxhash_32`
template <typename _Key, hash_algorithm _S = hash_algorithm::xxhash_32>
class hash;

template <typename _Key>
class hash<_Key, hash_algorithm::xxhash_32> : private __detail::_XXHash_32<_Key>
{
public:
  using __detail::_XXHash_32<_Key>::_XXHash_32;
  using __detail::_XXHash_32<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::xxhash_64> : private __detail::_XXHash_64<_Key>
{
public:
  using __detail::_XXHash_64<_Key>::_XXHash_64;
  using __detail::_XXHash_64<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_32> : private __detail::_MurmurHash3_32<_Key>
{
public:
  using __detail::_MurmurHash3_32<_Key>::_MurmurHash3_32;
  using __detail::_MurmurHash3_32<_Key>::operator();
};

#if _CCCL_HAS_INT128()

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x86_128> : private __detail::_MurmurHash3_x86_128<_Key>
{
public:
  using __detail::_MurmurHash3_x86_128<_Key>::_MurmurHash3_x86_128;
  using __detail::_MurmurHash3_x86_128<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x64_128> : private __detail::_MurmurHash3_x64_128<_Key>
{
public:
  using __detail::_MurmurHash3_x64_128<_Key>::_MurmurHash3_x64_128;
  using __detail::_MurmurHash3_x64_128<_Key>::operator();
};

#endif // _CCCL_HAS_INT128()
} // namespace cuda::experimental::cuco

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_HASH_FUNCTIONS_CUH
