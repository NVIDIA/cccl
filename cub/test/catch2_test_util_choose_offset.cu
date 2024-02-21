/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/detail/choose_offset.cuh>

#include <cuda/std/type_traits>

#include "catch2_test_helper.h"

CUB_TEST("Tests choose_offset", "[util][type]")
{
  // Uses unsigned 32-bit type for signed 32-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int32_t>, std::uint32_t>::value);

  // Uses unsigned 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int8_t>, std::uint32_t>::value);

  // Uses unsigned 64-bit type for signed 64-bit type
  STATIC_REQUIRE(::cuda::std::is_same<cub::detail::choose_offset_t<std::int64_t>, unsigned long long>::value);
}

CUB_TEST("Tests promote_small_offset", "[util][type]")
{
  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int32_t>, std::int32_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint32_t>, std::uint32_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint64_t>, std::uint64_t>::value);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int64_t>, std::int64_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int8_t>, std::int32_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::int16_t>, std::int32_t>::value);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(
    ::cuda::std::is_same<typename cub::detail::promote_small_offset_t<std::uint16_t>, std::int32_t>::value);
}
