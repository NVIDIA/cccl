/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#include <cuda/std/cstddef>

#include <cuda/experimental/__cuco/detail/hash_functions/utils.cuh>
#include <cuda/experimental/__cuco/extent.cuh>

#include <cstdint>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{

/**
 * @brief A `XXHash_32` hash function to hash the given argument on host and device.
 *
 * XXHash_32 implementation from
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
template <typename Key>
struct XXHash_32
{
private:
  static constexpr std::uint32_t prime1 = 0x9e3779b1u;
  static constexpr std::uint32_t prime2 = 0x85ebca77u;
  static constexpr std::uint32_t prime3 = 0xc2b2ae3du;
  static constexpr std::uint32_t prime4 = 0x27d4eb2fu;
  static constexpr std::uint32_t prime5 = 0x165667b1u;

public:
  using argument_type = Key; ///< The type of the values taken as argument
  using result_type   = std::uint32_t; ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  _CCCL_HOST_DEVICE constexpr XXHash_32(std::uint32_t seed = 0)
      : seed_{seed}
  {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr result_type _CCCL_HOST_DEVICE operator()(Key const& key) const noexcept
  {
    if constexpr (sizeof(Key) <= 16)
    {
      Key const key_copy = key;
      return compute_hash(reinterpret_cast<::cuda::std::byte const*>(&key_copy),
                          cuco::extent<std::size_t, sizeof(Key)>{});
    }
    else
    {
      return compute_hash(reinterpret_cast<::cuda::std::byte const*>(&key), cuco::extent<std::size_t, sizeof(Key)>{});
    }
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @tparam Extent The extent type
   *
   * @param bytes The input argument to hash
   * @param size The extent of the data in bytes
   * @return The resulting hash value
   */
  template <typename Extent>
  constexpr result_type _CCCL_HOST_DEVICE compute_hash(::cuda::std::byte const* bytes, Extent size) const noexcept
  {
    std::size_t offset = 0;
    std::uint32_t h32;

    // data can be processed in 16-byte chunks
    if (size >= 16)
    {
      auto const limit = size - 16;
      std::uint32_t v1 = seed_ + prime1 + prime2;
      std::uint32_t v2 = seed_ + prime2;
      std::uint32_t v3 = seed_;
      std::uint32_t v4 = seed_ - prime1;

      do
      {
        // pipeline 4*4byte computations
        auto const pipeline_offset = offset / 4;
        v1 += load_chunk<std::uint32_t>(bytes, pipeline_offset + 0) * prime2;
        v1 = rotl32(v1, 13);
        v1 *= prime1;
        v2 += load_chunk<std::uint32_t>(bytes, pipeline_offset + 1) * prime2;
        v2 = rotl32(v2, 13);
        v2 *= prime1;
        v3 += load_chunk<std::uint32_t>(bytes, pipeline_offset + 2) * prime2;
        v3 = rotl32(v3, 13);
        v3 *= prime1;
        v4 += load_chunk<std::uint32_t>(bytes, pipeline_offset + 3) * prime2;
        v4 = rotl32(v4, 13);
        v4 *= prime1;
        offset += 16;
      } while (offset <= limit);

      h32 = rotl32(v1, 1) + rotl32(v2, 7) + rotl32(v3, 12) + rotl32(v4, 18);
    }
    else
    {
      h32 = seed_ + prime5;
    }

    h32 += size;

    // remaining data can be processed in 4-byte chunks
    if ((size % 16) >= 4)
    {
      for (; offset <= size - 4; offset += 4)
      {
        h32 += load_chunk<std::uint32_t>(bytes, offset / 4) * prime3;
        h32 = rotl32(h32, 17) * prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if (size % 4)
    {
      while (offset < size)
      {
        h32 += (::cuda::std::to_integer<std::uint32_t>(bytes[offset]) & 255) * prime5;
        h32 = rotl32(h32, 11) * prime1;
        ++offset;
      }
    }

    return finalize(h32);
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `result_type`.
   *
   * @note This API is to ensure backward compatibility with existing use cases using `std::byte`.
   * Users are encouraged to use the appropriate `cuda::std::byte` overload whenever possible for
   * better support and performance on the device.
   *
   * @tparam Extent The extent type
   *
   * @param bytes The input argument to hash
   * @param size The extent of the data in bytes
   * @return The resulting hash value
   */
  template <typename Extent>
  constexpr result_type _CCCL_HOST_DEVICE compute_hash(::std::byte const* bytes, Extent size) const noexcept
  {
    return this->compute_hash(reinterpret_cast<::cuda::std::byte const*>(bytes), size);
  }

private:
  // avalanche helper
  constexpr _CCCL_HOST_DEVICE std::uint32_t finalize(std::uint32_t h) const noexcept
  {
    h ^= h >> 15;
    h *= prime2;
    h ^= h >> 13;
    h *= prime3;
    h ^= h >> 16;
    return h;
  }

  std::uint32_t seed_;
};

} // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_XXHASH_CUH
