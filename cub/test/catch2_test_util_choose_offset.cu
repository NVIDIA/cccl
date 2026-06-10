// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/choose_offset.cuh>

#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

C2H_TEST("Tests choose_offset", "[util][type]")
{
  // Uses unsigned 32-bit type for signed 32-bit type
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_offset_t<std::int32_t>, std::uint32_t>);

  // Uses unsigned 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_offset_t<std::int8_t>, std::uint32_t>);

  // Uses unsigned 64-bit type for signed 64-bit type
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_offset_t<std::int64_t>, unsigned long long>);
}

C2H_TEST("Tests choose_signed_offset", "[util][type]")
{
  // Uses signed 64-bit type for unsigned signed 32-bit type
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_signed_offset_t<std::uint32_t>, std::int64_t>);

  // Uses signed 32-bit type for signed 32-bit type
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_signed_offset_t<std::int32_t>, std::int32_t>);

  // Uses signed 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_signed_offset_t<std::int8_t>, std::int32_t>);

  // Uses signed 64-bit type for signed 64-bit type
  STATIC_REQUIRE(cuda::std::is_same_v<cub::detail::choose_signed_offset_t<std::int64_t>, std::int64_t>);

  // Offset type covers maximum number representable by a signed 32-bit integer
  REQUIRE(cudaSuccess
          == cub::detail::choose_signed_offset<std::int32_t>::is_exceeding_offset_type(
            cuda::std::numeric_limits<std::int32_t>::max()));

  // Offset type covers maximum number representable by a signed 64-bit integer
  REQUIRE(cudaSuccess
          == cub::detail::choose_signed_offset<std::int64_t>::is_exceeding_offset_type(
            cuda::std::numeric_limits<std::int64_t>::max()));

  // Offset type does not support maximum number representable by an unsigned 64-bit integer
  REQUIRE(cudaErrorInvalidValue
          == cub::detail::choose_signed_offset<std::uint64_t>::is_exceeding_offset_type(
            cuda::std::numeric_limits<std::uint64_t>::max()));
}

C2H_TEST("Tests promote_small_offset", "[util][type]")
{
  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::int32_t>, std::int32_t>);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::uint32_t>, std::uint32_t>);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::uint64_t>, std::uint64_t>);

  // Uses input type for types of at least 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::int64_t>, std::int64_t>);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::int8_t>, std::int32_t>);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::int16_t>, std::int32_t>);

  // Uses 32-bit type for type smaller than 32 bits
  STATIC_REQUIRE(cuda::std::is_same_v<typename cub::detail::promote_small_offset_t<std::uint16_t>, std::int32_t>);
}
