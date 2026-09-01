//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/*
 * `__xxhash_32` and `__xxhash_64` implementation from
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

#ifndef _CUDA___FUNCTIONAL_HASH_XXHASH_H
#define _CUDA___FUNCTIONAL_HASH_XXHASH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/hash/utils.h>
#include <cuda/__utility/static_for.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__bit/rotl.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
//! @brief A `__xxhash_32` hash function to hash the given argument on host and device.
//!
//! @tparam _Key The type of the values to hash
template <typename _Key>
struct __xxhash_32
{
private:
  static constexpr ::cuda::std::uint32_t __prime1 = 0x9e3779b1u;
  static constexpr ::cuda::std::uint32_t __prime2 = 0x85ebca77u;
  static constexpr ::cuda::std::uint32_t __prime3 = 0xc2b2ae3du;
  static constexpr ::cuda::std::uint32_t __prime4 = 0x27d4eb2fu;
  static constexpr ::cuda::std::uint32_t __prime5 = 0x165667b1u;

  static constexpr ::cuda::std::uint32_t __block_size = 4;
  static constexpr ::cuda::std::uint32_t __chunk_size = 16;

  struct __compute_holder_block
  {
    template <class _Index, class _Holder>
    _CCCL_HOST_DEVICE_API constexpr void operator()(
      _Index __i,
      const _Holder& __holder,
      ::cuda::std::array<::cuda::std::uint32_t, 4>& __v,
      ::cuda::std::uint32_t& __offset) const noexcept
    {
      __v[__i] += __holder.__blocks_[__offset++] * __prime2;
      __v[__i] = ::cuda::std::rotl(__v[__i], 13);
      __v[__i] *= __prime1;
    }
  };

  struct __compute_span_block
  {
    template <class _Index>
    _CCCL_HOST_DEVICE_API constexpr void operator()(
      _Index __i,
      const ::cuda::std::byte* __bytes,
      ::cuda::std::size_t __pipeline_offset,
      ::cuda::std::array<::cuda::std::uint32_t, 4>& __v) const noexcept
    {
      __v[__i] += ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, __pipeline_offset + __i) * __prime2;
      __v[__i] = ::cuda::std::rotl(__v[__i], 13);
      __v[__i] *= __prime1;
    }
  };

public:
  //! @brief Constructs a XXH32 hash function with the given `seed`.
  //! @param __seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE_API constexpr explicit __xxhash_32(::cuda::std::uint32_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //! @param __key The input argument to hash
  //! @return The resulting hash value for `__key`
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t operator()(const _Key& __key) const noexcept
  {
    using _Holder _CCCL_NODEBUG = __byte_holder<sizeof(_Key), __chunk_size, __block_size, true, ::cuda::std::uint32_t>;
    // Materialize a copy so the device compiler can use wide loads.
    const _Key __copy{__key};
    return __compute_hash(::cuda::std::bit_cast<_Holder>(__copy));
  }

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //! @tparam _Extent The extent type
  //! @param __keys span of keys to hash
  //! @return The resulting hash value
  template <::cuda::std::size_t _Extent>
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint32_t
  operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    return __compute_hash_span(__keys);
  }

private:
  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //!
  //! @tparam _Holder The byte holder type
  //! @param __holder The input argument to hash in form of a byte holder
  //! @return The resulting hash value
  template <class _Holder>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t __compute_hash(_Holder __holder) const noexcept
  {
    [[maybe_unused]] ::cuda::std::uint32_t __offset = 0;
    ::cuda::std::uint32_t __h32                     = {};

    // process data in 16-byte chunks
    if constexpr (_Holder::__num_chunks > 0)
    {
      ::cuda::std::array<::cuda::std::uint32_t, 4> __v;
      __v[0] = __seed_ + __prime1 + __prime2;
      __v[1] = __seed_ + __prime2;
      __v[2] = __seed_;
      __v[3] = __seed_ - __prime1;

      for (::cuda::std::uint32_t __chunk = 0; __chunk < _Holder::__num_chunks; ++__chunk)
      {
        ::cuda::static_for<4>(__compute_holder_block{}, __holder, __v, __offset);
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
        __h32 += __holder.__blocks_[__offset] * __prime3;
        __h32 = ::cuda::std::rotl(__h32, 17) * __prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if constexpr (_Holder::__tail_size > 0)
    {
      for (::cuda::std::uint32_t __i = 0; __i < _Holder::__tail_size; ++__i)
      {
        __h32 += (static_cast<::cuda::std::uint32_t>(__holder.__bytes_[__i])) * __prime5;
        __h32 = ::cuda::std::rotl(__h32, 11) * __prime1;
      }
    }

    return __finalize(__h32);
  }

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint32_t`.
  //!
  //! @param __keys The span of keys to hash
  //! @return The resulting hash value
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint32_t
  __compute_hash_span(::cuda::std::span<_Key> __keys) const noexcept
  {
    const auto __bytes = ::cuda::std::as_bytes(__keys).data();
    const auto __size  = __keys.size_bytes();

    ::cuda::std::size_t __offset = 0;
    ::cuda::std::uint32_t __h32  = {};

    // data can be processed in 16-byte chunks
    if (__size >= 16)
    {
      const auto __limit = __size - 16;
      ::cuda::std::array<::cuda::std::uint32_t, 4> __v;

      __v[0] = __seed_ + __prime1 + __prime2;
      __v[1] = __seed_ + __prime2;
      __v[2] = __seed_;
      __v[3] = __seed_ - __prime1;

      for (; __offset <= __limit; __offset += 16)
      {
        // pipeline 4*4byte computations
        const auto __pipeline_offset = __offset / 4;
        ::cuda::static_for<4>(__compute_span_block{}, __bytes, __pipeline_offset, __v);
      }

      __h32 = ::cuda::std::rotl(__v[0], 1) + ::cuda::std::rotl(__v[1], 7) + ::cuda::std::rotl(__v[2], 12)
            + ::cuda::std::rotl(__v[3], 18);
    }
    else
    {
      __h32 = __seed_ + __prime5;
    }

    __h32 += static_cast<::cuda::std::uint32_t>(__size);

    // remaining data can be processed in 4-byte chunks
    if ((__size % 16) >= 4)
    {
      for (; __offset <= __size - 4; __offset += 4)
      {
        __h32 += ::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, __offset / 4) * __prime3;
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

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t
  __finalize(::cuda::std::uint32_t __h) const noexcept
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

//! @brief A `XXHash_64` hash function to hash the given argument on host and device.
//!
//! @tparam _Key The type of the values to hash
template <typename _Key>
struct __xxhash_64
{
private:
  static constexpr ::cuda::std::uint64_t __prime1 = 11400714785074694791ull;
  static constexpr ::cuda::std::uint64_t __prime2 = 14029467366897019727ull;
  static constexpr ::cuda::std::uint64_t __prime3 = 1609587929392839161ull;
  static constexpr ::cuda::std::uint64_t __prime4 = 9650029242287828579ull;
  static constexpr ::cuda::std::uint64_t __prime5 = 2870177450012600261ull;

  struct __compute_span_block
  {
    template <class _Index>
    _CCCL_HOST_DEVICE_API constexpr void operator()(
      _Index __i,
      const ::cuda::std::byte* __bytes,
      ::cuda::std::size_t __pipeline_offset,
      ::cuda::std::array<::cuda::std::uint64_t, 4>& __v) const noexcept
    {
      __v[__i] += ::cuda::__load_chunk<::cuda::std::uint64_t>(__bytes, __pipeline_offset + __i) * __prime2;
      __v[__i] = ::cuda::std::rotl(__v[__i], 31);
      __v[__i] *= __prime1;
    }
  };

  struct __merge_accumulator
  {
    template <class _Index>
    _CCCL_HOST_DEVICE_API constexpr void operator()(
      _Index __i, ::cuda::std::array<::cuda::std::uint64_t, 4>& __v, ::cuda::std::uint64_t& __h64) const noexcept
    {
      __v[__i] *= __prime2;
      __v[__i] = ::cuda::std::rotl(__v[__i], 31);
      __v[__i] *= __prime1;
      __h64 ^= __v[__i];
      __h64 = __h64 * __prime1 + __prime4;
    }
  };

public:
  //! @brief Constructs a XXH64 hash function with the given `seed`.
  //!
  //! @param __seed A custom number to randomize the resulting hash value
  _CCCL_HOST_DEVICE_API constexpr explicit __xxhash_64(::cuda::std::uint64_t __seed = 0)
      : __seed_{__seed}
  {}

  //! @brief Returns a hash value for its argument, as a value of type `result_type`.
  //!
  //! @param __key The input argument to hash
  //! @return The resulting hash value for `__key`
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t operator()(const _Key& __key) const noexcept
  {
    if constexpr (sizeof(_Key) <= 16)
    {
      // Materialize a copy so the device compiler can use wide loads.
      const _Key __copy{__key};
      return __compute_hash_span(::cuda::std::span<const _Key, 1>{&__copy, 1});
    }
    else
    {
      return __compute_hash_span(::cuda::std::span<const _Key, 1>{&__key, 1});
    }
  }

  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint64_t`.
  //!
  //! @tparam _Extent The extent type
  //! @param __keys span of keys to hash
  //! @return The resulting hash value
  template <::cuda::std::size_t _Extent>
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t
  operator()(::cuda::std::span<_Key, _Extent> __keys) const noexcept
  {
    return __compute_hash_span(__keys);
  }

private:
  //! @brief Returns a hash value for its argument, as a value of type `::cuda::std::uint64_t`.
  //!
  //! @param __keys The span of keys to hash
  //! @return The resulting hash value
  [[nodiscard]] _CCCL_HOST_DEVICE_API ::cuda::std::uint64_t
  __compute_hash_span(::cuda::std::span<const _Key> __keys) const noexcept
  {
    const auto __bytes = ::cuda::std::as_bytes(__keys).data();
    const auto __size  = __keys.size_bytes();

    ::cuda::std::size_t __offset = 0;
    ::cuda::std::uint64_t __h64  = {};

    // process data in 32-byte chunks
    if (__size >= 32)
    {
      const auto __limit = __size - 32;
      ::cuda::std::array<::cuda::std::uint64_t, 4> __v;

      __v[0] = __seed_ + __prime1 + __prime2;
      __v[1] = __seed_ + __prime2;
      __v[2] = __seed_;
      __v[3] = __seed_ - __prime1;

      for (; __offset <= __limit; __offset += 32)
      {
        // pipeline 4*8byte computations
        const auto __pipeline_offset = __offset / 8;
        ::cuda::static_for<4>(__compute_span_block{}, __bytes, __pipeline_offset, __v);
      }

      __h64 = ::cuda::std::rotl(__v[0], 1) + ::cuda::std::rotl(__v[1], 7) + ::cuda::std::rotl(__v[2], 12)
            + ::cuda::std::rotl(__v[3], 18);

      ::cuda::static_for<4>(__merge_accumulator{}, __v, __h64);
    }
    else
    {
      __h64 = __seed_ + __prime5;
    }

    __h64 += __size;

    // remaining data can be processed in 8-byte chunks
    if ((__size % 32) >= 8)
    {
      for (; __offset <= __size - 8; __offset += 8)
      {
        ::cuda::std::uint64_t __k1 = ::cuda::__load_chunk<::cuda::std::uint64_t>(__bytes, __offset / 8) * __prime2;
        __k1                       = ::cuda::std::rotl(__k1, 31) * __prime1;
        __h64 ^= __k1;
        __h64 = ::cuda::std::rotl(__h64, 27) * __prime1 + __prime4;
      }
    }

    // remaining data can be processed in 4-byte chunks
    if ((__size % 8) >= 4)
    {
      for (; __offset <= __size - 4; __offset += 4)
      {
        __h64 ^= (::cuda::__load_chunk<::cuda::std::uint32_t>(__bytes, __offset / 4)) * __prime1;
        __h64 = ::cuda::std::rotl(__h64, 23) * __prime2 + __prime3;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of a previous
    // block size
    if (__size % 4)
    {
      while (__offset < __size)
      {
        __h64 ^= (::cuda::std::to_integer<::cuda::std::uint32_t>(__bytes[__offset])) * __prime5;
        __h64 = ::cuda::std::rotl(__h64, 11) * __prime1;
        ++__offset;
      }
    }
    return __finalize(__h64);
  }

  // avalanche helper
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint64_t
  __finalize(::cuda::std::uint64_t __h) const noexcept
  {
    __h ^= __h >> 33;
    __h *= __prime2;
    __h ^= __h >> 29;
    __h *= __prime3;
    __h ^= __h >> 32;
    return __h;
  }

  ::cuda::std::uint64_t __seed_;
};
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_HASH_XXHASH_H
