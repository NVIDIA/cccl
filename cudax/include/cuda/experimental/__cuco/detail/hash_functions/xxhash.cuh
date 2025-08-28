//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
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
 */

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

#include <cuda/__utility/static_for.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__cuco/detail/hash_functions/utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{

//! @brief A `_XXHash_32` hash function to hash the given argument on host and device.
//!
//! @tparam Key The type of the values to hash
template <typename _Key>
struct _XXHash_32
{
private:
  static constexpr ::cuda::std::uint32_t __prime1 = 0x9e3779b1u;
  static constexpr ::cuda::std::uint32_t __prime2 = 0x85ebca77u;
  static constexpr ::cuda::std::uint32_t __prime3 = 0xc2b2ae3du;
  static constexpr ::cuda::std::uint32_t __prime4 = 0x27d4eb2fu;
  static constexpr ::cuda::std::uint32_t __prime5 = 0x165667b1u;

  static constexpr size_t __block_size = 4;
  static constexpr size_t __chunk_size = 16;

  //! @brief Type erased holder of all the bytes
  template <size_t _KeySize,
            size_t _Alignment,
            bool _HasChunks = (_KeySize >= __block_size),
            bool _HasTail   = (_KeySize % __block_size)>
  struct alignas(_Alignment) _Byte_holder
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of uint32_t blocks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    //! The number of 16-byte chunks
    static constexpr size_t __num_chunks = _KeySize / __chunk_size;

    alignas(_Alignment)::cuda::std::uint32_t __blocks[__num_blocks];
    unsigned char __bytes[__tail_size];
  };

  //! @brief Type erased holder of small types < __block_size
  template <size_t _KeySize, size_t _Alignment>
  struct alignas(_Alignment) _Byte_holder<_KeySize, _Alignment, false, true>
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of uint32_t blocks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    //! The number of 16-byte chunks
    static constexpr size_t __num_chunks = _KeySize / __chunk_size;

    alignas(_Alignment) unsigned char __bytes[__tail_size];
  };

  //! @brief Type erased holder of types without trailing bytes
  template <size_t _KeySize, size_t _Alignment>
  struct alignas(_Alignment) _Byte_holder<_KeySize, _Alignment, true, false>
  {
    //! The number of trailing bytes that do not fit into a uint32_t
    static constexpr size_t __tail_size = _KeySize % __block_size;

    //! The number of uint32_t blocks
    static constexpr size_t __num_blocks = _KeySize / __block_size;

    //! The number of 16-byte chunks
    static constexpr size_t __num_chunks = _KeySize / __chunk_size;

    alignas(_Alignment)::cuda::std::uint32_t __blocks[__num_blocks];
  };

public:
  //! @brief Constructs a XXH32 hash function with the given `seed`.
  //! @param seed A custom number to randomize the resulting hash value
  _CCCL_API constexpr _XXHash_32(::cuda::std::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //! @param __key The input argument to hash
  //! @return The resulting hash value for `__key`
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t operator()(_Key const& __key) const noexcept
  {
    using _Holder = _Byte_holder<sizeof(_Key), alignof(_Key)>;
    // explicit copy to avoid emitting a bunch of LDG.8 instructions
    const _Key __copy{__key};
    return __compute_hash(::cuda::std::bit_cast<_Holder>(__copy));
  }

  template <size_t _Extent>
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t
  operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    // TODO: optimize when _Extent is known at compile time i.e
    // _Extent != ::cuda::std::dynamic_extent, dispatch to bit_cast based implementation
    return __compute_hash_span(__keys);
  }

private:
  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //!
  //! @tparam _Extent The extent type
  //!
  //! @return The resulting hash value
  template <class _Holder>
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t __compute_hash(_Holder __holder) const noexcept
  {
    size_t __offset             = 0;
    ::cuda::std::uint32_t __h32 = {};

    // process data in 16-byte chunks
    if constexpr (_Holder::__num_chunks > 0)
    {
      ::cuda::std::array<::cuda::std::uint32_t, 4> __v;
      __v[0] = __seed_ + __prime1 + __prime2;
      __v[1] = __seed_ + __prime2;
      __v[2] = __seed_;
      __v[3] = __seed_ - __prime1;

      for (size_t __i = 0; __i < _Holder::__num_chunks; ++__i)
      {
        cuda::static_for<4>([&](auto i) {
          __v[i] += __holder.__blocks[__offset++] * __prime2;
          __v[i] = ::cuda::std::rotl(__v[i], 13);
          __v[i] *= __prime1;
        });
      }
      __h32 = ::cuda::std::rotl(__v[0], 1) + ::cuda::std::rotl(__v[1], 7) + ::cuda::std::rotl(__v[2], 12)
            + ::cuda::std::rotl(__v[3], 18);
    }
    else
    {
      __h32 = __seed_ + __prime5;
    }

    __h32 += ::cuda::std::uint32_t{sizeof(_Holder)};

    // remaining data can be processed in 4-byte chunks
    if constexpr (_Holder::__num_blocks % __chunk_size > 0)
    {
      for (; __offset < _Holder::__num_blocks; ++__offset)
      {
        __h32 += __holder.__blocks[__offset] * __prime3;
        __h32 = ::cuda::std::rotl(__h32, 17) * __prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if constexpr (_Holder::__tail_size > 0)
    {
      for (size_t __i = 0; __i < _Holder::__tail_size; ++__i)
      {
        __h32 += (static_cast<::cuda::std::uint32_t>(__holder.__bytes[__i]) & 255) * __prime5;
        __h32 = ::cuda::std::rotl(__h32, 11) * __prime1;
      }
    }

    return __finalize(__h32);
  }

  [[nodiscard]] _CCCL_API ::cuda::std::uint32_t __compute_hash_span(::cuda::std::span<_Key> __keys) const noexcept
  {
    auto __bytes      = ::cuda::std::as_bytes(__keys).data();
    auto const __size = __keys.size_bytes();

    size_t __offset             = 0;
    ::cuda::std::uint32_t __h32 = {};

    // data can be processed in 16-byte chunks
    if (__size >= 16)
    {
      auto const __limit = __size - 16;
      ::cuda::std::array<::cuda::std::uint32_t, 4> __v;

      __v[0] = __seed_ + __prime1 + __prime2;
      __v[1] = __seed_ + __prime2;
      __v[2] = __seed_;
      __v[3] = __seed_ - __prime1;

      for (; __offset <= __limit; __offset += 16)
      {
        // pipeline 4*4byte computations
        auto const __pipeline_offset = __offset / 4;
        cuda::static_for<4>([&](auto i) {
          __v[i] += __load_chunk<::cuda::std::uint32_t>(__bytes, __pipeline_offset + i) * __prime2;
          __v[i] = ::cuda::std::rotl(__v[i], 13);
          __v[i] *= __prime1;
        });
      }

      __h32 = ::cuda::std::rotl(__v[0], 1) + ::cuda::std::rotl(__v[1], 7) + ::cuda::std::rotl(__v[2], 12)
            + ::cuda::std::rotl(__v[3], 18);
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
        __h32 += __load_chunk<::cuda::std::uint32_t>(__bytes, __offset / 4) * __prime3;
        __h32 = ::cuda::std::rotl(__h32, 17) * __prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if (__size % 4)
    {
      while (__offset < __size)
      {
        __h32 += (::cuda::std::to_integer<::cuda::std::uint32_t>(__bytes[__offset]) & 255) * __prime5;
        __h32 = ::cuda::std::rotl(__h32, 11) * __prime1;
        ++__offset;
      }
    }

    return __finalize(__h32);
  }

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::uint32_t __finalize(::cuda::std::uint32_t __h) const noexcept
  {
    __h ^= __h >> 15;
    __h *= __prime2;
    __h ^= __h >> 13;
    __h *= __prime3;
    __h ^= __h >> 16;
    return __h;
  }

  ::cuda::std::uint32_t __seed_;
};

} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH
