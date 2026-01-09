// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/util_math.cuh>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>

C2H_TEST("Tests safe_add_bound_to_max", "[util][math]")
{
  REQUIRE(cub::detail::safe_add_bound_to_max(0U, cuda::std::numeric_limits<std::uint32_t>::max())
          == cuda::std::numeric_limits<std::uint32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(cuda::std::numeric_limits<std::uint32_t>::max(), 0U)
          == cuda::std::numeric_limits<std::uint32_t>::max());

  // We do not overflow
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int32_t{0}, cuda::std::numeric_limits<std::int32_t>::max())
          == cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(cuda::std::numeric_limits<std::int32_t>::max(), std::int32_t{0})
          == cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int32_t{1}, cuda::std::numeric_limits<std::int32_t>::max())
          == cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(cuda::std::numeric_limits<std::int32_t>::max(), std::int32_t{1})
          == cuda::std::numeric_limits<std::int32_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(
            cuda::std::numeric_limits<std::int32_t>::max(), cuda::std::numeric_limits<std::int32_t>::max())
          == cuda::std::numeric_limits<std::int32_t>::max());

  // We do not overflow
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int64_t{0}, cuda::std::numeric_limits<std::int64_t>::max())
          == cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(cuda::std::numeric_limits<std::int64_t>::max(), std::int64_t{0LL})
          == cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(std::int64_t{1LL}, cuda::std::numeric_limits<std::int64_t>::max())
          == cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(cuda::std::numeric_limits<std::int64_t>::max(), std::int64_t{1LL})
          == cuda::std::numeric_limits<std::int64_t>::max());
  REQUIRE(cub::detail::safe_add_bound_to_max(
            cuda::std::numeric_limits<std::int64_t>::max(), cuda::std::numeric_limits<std::int64_t>::max())
          == cuda::std::numeric_limits<std::int64_t>::max());

  // We do not underflow for negative rhs (not, lhs must not be negative per documentation)
  REQUIRE(cub::detail::safe_add_bound_to_max(0, -1) == -1);
  REQUIRE(cub::detail::safe_add_bound_to_max(1, -1) == 0);
}
