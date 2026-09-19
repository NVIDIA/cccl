//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_HASH_H
#define _CUDA___FUNCTIONAL_HASH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/hash/murmurhash3.h>
#include <cuda/__functional/hash/xxhash.h>
#include <cuda/std/__type_traits/always_false.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Hash algorithms supported by `cuda::hash`.
enum class hash_algorithm
{
  xxhash_32,
  xxhash_64,
  murmurhash3_32,
  murmurhash3_x86_128,
  murmurhash3_x64_128
};

//! @brief A hash function class specialized for different hash algorithms.
//!
//! @tparam _Key The type of the values to hash
//! @tparam _Algorithm The hash algorithm to use, defaults to `hash_algorithm::xxhash_64`
template <typename _Key, hash_algorithm _Algorithm = hash_algorithm::xxhash_64>
class hash;

template <typename _Key>
class hash<_Key, hash_algorithm::xxhash_32> : private ::cuda::__xxhash_32<_Key>
{
public:
  using ::cuda::__xxhash_32<_Key>::__xxhash_32;
  using ::cuda::__xxhash_32<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::xxhash_64> : private ::cuda::__xxhash_64<_Key>
{
public:
  using ::cuda::__xxhash_64<_Key>::__xxhash_64;
  using ::cuda::__xxhash_64<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_32> : private ::cuda::__murmurhash3_32<_Key>
{
public:
  using ::cuda::__murmurhash3_32<_Key>::__murmurhash3_32;
  using ::cuda::__murmurhash3_32<_Key>::operator();
};

#if _CCCL_HAS_INT128()

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x86_128> : private ::cuda::__murmurhash3_x86_128<_Key>
{
public:
  using ::cuda::__murmurhash3_x86_128<_Key>::__murmurhash3_x86_128;
  using ::cuda::__murmurhash3_x86_128<_Key>::operator();
};

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x64_128> : private ::cuda::__murmurhash3_x64_128<_Key>
{
public:
  using ::cuda::__murmurhash3_x64_128<_Key>::__murmurhash3_x64_128;
  using ::cuda::__murmurhash3_x64_128<_Key>::operator();
};

#else // _CCCL_HAS_INT128()

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x86_128>
{
  static_assert(::cuda::std::__always_false_v<_Key>,
                "cuda::hash with hash_algorithm::murmurhash3_x86_128 requires compiler support for __int128");
};

template <typename _Key>
class hash<_Key, hash_algorithm::murmurhash3_x64_128>
{
  static_assert(::cuda::std::__always_false_v<_Key>,
                "cuda::hash with hash_algorithm::murmurhash3_x64_128 requires compiler support for __int128");
};

#endif // _CCCL_HAS_INT128()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_HASH_H
