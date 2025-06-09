//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH
#define _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/detail/hash_functions/utils.cuh>
#include <cuda/experimental/__cuco/extent.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{

/**
 * @brief A `_XXHash_32` hash function to hash the given argument on host and device.
 *
 * _XXHash_32 implementation from
 * https://github.com/Cyan4973/xxHash
 * -----------------------------------------------------------------------------
 * xxHash - Extremely Fast Hash algorithm
 * Header File
 * Copyright (C) 2012-2021 Yann Collet
 *
 * BSD 2-Clause License (https://www.opensource.org/licenses/bsd-license.php)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above
 *      copyright notice, this list of conditions and the following disclaimer
 *      in the documentation and/or other materials provided with the
 *      distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @tparam Key The type of the values to hash
 */
template <typename _Key>
struct _XXHash_32
{
private:
  static constexpr _CUDA_VSTD::uint32_t __prime1 = 0x9e3779b1u;
  static constexpr _CUDA_VSTD::uint32_t __prime2 = 0x85ebca77u;
  static constexpr _CUDA_VSTD::uint32_t __prime3 = 0xc2b2ae3du;
  static constexpr _CUDA_VSTD::uint32_t __prime4 = 0x27d4eb2fu;
  static constexpr _CUDA_VSTD::uint32_t __prime5 = 0x165667b1u;

public:
  //! @brief Constructs a XXH32 hash function with the given `seed`.
  //! @param seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE constexpr _XXHash_32(_CUDA_VSTD::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a hash value for its argument, as a value of type `_CUDA_VSTD::uint32_t`.

  //! @param __key The input argument to hash
  //! @return The resulting hash value for `__key`
  [[nodiscard]] constexpr _CUDA_VSTD::uint32_t _CCCL_HOST_DEVICE operator()(_Key const& __key) const noexcept
  {
    if constexpr (sizeof(_Key) <= 16)
    {
      _Key const __key_copy = __key;
      return __compute_hash(reinterpret_cast<::cuda::std::byte const*>(&__key_copy),
                            cuco::extent<_CUDA_VSTD::uint32_t, sizeof(_Key)>{});
    }
    else
    {
      return __compute_hash(reinterpret_cast<::cuda::std::byte const*>(&__key),
                            cuco::extent<_CUDA_VSTD::uint32_t, sizeof(_Key)>{});
    }
  }

  //! @brief Returns a hash value for its argument, as a value of type `_CUDA_VSTD::uint32_t`.
  //!
  //! @tparam _Extent The extent type
  //!
  //! @param __bytes The input argument to hash
  //! @param __size The extent of the data in bytes
  //! @return The resulting hash value
  template <typename _Extent>
  [[nodiscard]] constexpr _CUDA_VSTD::uint32_t _CCCL_HOST_DEVICE
  __compute_hash(::cuda::std::byte const* __bytes, _Extent __size) const noexcept
  {
    _CUDA_VSTD::size_t __offset = 0;
    _CUDA_VSTD::uint32_t __h32;

    // data can be processed in 16-byte chunks
    if (__size >= 16)
    {
      auto const __limit        = __size - 16;
      _CUDA_VSTD::uint32_t __v1 = __seed_ + __prime1 + __prime2;
      _CUDA_VSTD::uint32_t __v2 = __seed_ + __prime2;
      _CUDA_VSTD::uint32_t __v3 = __seed_;
      _CUDA_VSTD::uint32_t __v4 = __seed_ - __prime1;

      do
      {
        // pipeline 4*4byte computations
        auto const __pipeline_offset = __offset / 4;
        __v1 += __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __pipeline_offset + 0) * __prime2;
        __v1 = ::cuda::std::rotl(__v1, 13);
        __v1 *= __prime1;
        __v2 += __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __pipeline_offset + 1) * __prime2;
        __v2 = ::cuda::std::rotl(__v2, 13);
        __v2 *= __prime1;
        __v3 += __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __pipeline_offset + 2) * __prime2;
        __v3 = ::cuda::std::rotl(__v3, 13);
        __v3 *= __prime1;
        __v4 += __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __pipeline_offset + 3) * __prime2;
        __v4 = ::cuda::std::rotl(__v4, 13);
        __v4 *= __prime1;
        __offset += 16;
      } while (__offset <= __limit);

      __h32 = ::cuda::std::rotl(__v1, 1) + ::cuda::std::rotl(__v2, 7) + ::cuda::std::rotl(__v3, 12)
            + ::cuda::std::rotl(__v4, 18);
    }
    else
    {
      __h32 = __seed_ + __prime5;
    }

    __h32 += __size;

    // remaining data can be processed in 4-byte chunks
    if ((__size % 16) >= 4)
    {
      for (; __offset <= __size - 4; __offset += 4)
      {
        __h32 += __load_chunk<_CUDA_VSTD::uint32_t>(__bytes, __offset / 4) * __prime3;
        __h32 = ::cuda::std::rotl(__h32, 17) * __prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if (__size % 4)
    {
      while (__offset < __size)
      {
        __h32 += (::cuda::std::to_integer<_CUDA_VSTD::uint32_t>(__bytes[__offset]) & 255) * __prime5;
        __h32 = ::cuda::std::rotl(__h32, 11) * __prime1;
        ++__offset;
      }
    }

    return __finalize(__h32);
  }

  //! @brief Returns a hash value for its argument, as a value of type `_CUDA_VSTD::uint32_t`.
  //!
  //! @note This API is to ensure backward compatibility with existing use cases using `std::byte`.
  //! Users are encouraged to use the appropriate `cuda::std::byte` overload whenever possible for
  //! better support and performance on the device.
  //!
  //! @tparam _Extent The extent type
  //!
  //! @param __bytes The input argument to hash
  //! @param __size The extent of the data in bytes
  //! @return The resulting hash value
  template <typename _Extent>
  [[nodiscard]] constexpr _CUDA_VSTD::uint32_t _CCCL_HOST_DEVICE
  __compute_hash(::std::byte const* __bytes, _Extent __size) const noexcept
  {
    return this->__compute_hash(reinterpret_cast<::cuda::std::byte const*>(__bytes), __size);
  }

private:
  // avalanche helper
  [[nodiscard]] constexpr _CCCL_HOST_DEVICE _CUDA_VSTD::uint32_t __finalize(_CUDA_VSTD::uint32_t __h) const noexcept
  {
    __h ^= __h >> 15;
    __h *= __prime2;
    __h ^= __h >> 13;
    __h *= __prime3;
    __h ^= __h >> 16;
    return __h;
  }

  _CUDA_VSTD::uint32_t __seed_;
};

} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH
