/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from cuCollections d0d2abe0bae7b18bbc5641c6a94fdf3a75171994.
 * Changes: standalone byte loads, static byte extent, XXH32 only, local namespace.
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
 */
#ifndef _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_XXHASH_H
#define _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_XXHASH_H

#include <cuda/std/detail/__config>
#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN
namespace __detail::__hashgraph_shuffle
{
// Seed-zero XXH32 for the generator's fixed eight-byte little-endian key.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::uint32_t __xxhash32(::cuda::std::uint64_t __key) noexcept
{
  ::cuda::std::uint32_t __hash = 0x165667b1U + 8U;
  for (unsigned __shift = 0; __shift < 64; __shift += 32)
  {
    __hash += static_cast<::cuda::std::uint32_t>(__key >> __shift) * 0xc2b2ae3dU;
    __hash = ((__hash << 17) | (__hash >> 15)) * 0x27d4eb2fU;
  }
  __hash = (__hash ^ (__hash >> 15)) * 0x85ebca77U;
  __hash = (__hash ^ (__hash >> 13)) * 0xc2b2ae3dU;
  return __hash ^ (__hash >> 16);
}
} // namespace __detail::__hashgraph_shuffle
_CCCL_END_NAMESPACE_CUDA_MGMN
// NOLINTEND(bugprone-reserved-identifier)
#include <cuda/std/__cccl/epilogue.h>
#endif // _CUDA_EXPERIMENTAL_MGMN___ALGORITHM_SHUFFLE_XXHASH_H
