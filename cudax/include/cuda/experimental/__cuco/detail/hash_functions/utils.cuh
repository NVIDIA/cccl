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

#ifndef _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
#define _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH

#include <cuda/__cccl_config>
#include <cuda/std/cstddef>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::detail
{

template <typename T, typename U, typename Extent>
constexpr _CCCL_HOST_DEVICE T load_chunk(U const* const data, Extent index) noexcept
{
  auto const bytes = reinterpret_cast<::cuda::std::byte const*>(data);
  T chunk;
  memcpy(&chunk, bytes + index * sizeof(T), sizeof(T));
  return chunk;
}

constexpr _CCCL_HOST_DEVICE std::uint32_t rotl32(std::uint32_t x, std::int8_t r) noexcept
{
  return (x << r) | (x >> (32 - r));
}

constexpr _CCCL_HOST_DEVICE std::uint64_t rotl64(std::uint64_t x, std::int8_t r) noexcept
{
  return (x << r) | (x >> (64 - r));
}

}; // namespace cuda::experimental::cuco::detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__CUCO_DETAIL_HASH_FUNCTIONS_UTILS_CUH
