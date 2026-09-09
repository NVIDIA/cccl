// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cuda/devices>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/stream>

#include <stdexcept>

#include "cub_test_macros.h"
#include <c2h/buffer_generators.cuh>
#include <c2h/generator_types.h>
#include <c2h/vector_generators.h>

CUB_TEST("c2h uniform offset size validation rejects invalid element counts", "[c2h][generators]", CUB_SMALL)
{
  REQUIRE(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{0}) == 2);
  REQUIRE(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{1}) == 3);

  REQUIRE_THROWS_AS(c2h::detail::checked_uniform_offsets_size(cuda::std::int32_t{-1}), std::invalid_argument);
  REQUIRE_THROWS_AS(c2h::detail::checked_uniform_offsets_size((cuda::std::numeric_limits<cuda::std::int32_t>::max)()),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(
    c2h::detail::checked_uniform_offsets_size((cuda::std::numeric_limits<cuda::std::uint64_t>::max)() - 1),
    std::invalid_argument);
}

CUB_TEST("c2h uniform offset generators validate sizes before allocation", "[c2h][generators]", CUB_SMALL)
{
  const auto seed = c2h::seed_t{0};

  REQUIRE_THROWS_AS(
    c2h::gen_uniform_offsets(seed, cuda::std::int32_t{-1}, cuda::std::int32_t{0}, cuda::std::int32_t{1}),
    std::invalid_argument);

  const auto stream = cuda::stream_ref{cudaStream_t{}};
  const auto device = cuda::device_ref{0};
  REQUIRE_THROWS_AS(c2h::gen_uniform_offsets_device_buffer(
                      stream, device, seed, cuda::std::int32_t{-1}, cuda::std::int32_t{0}, cuda::std::int32_t{1}),
                    std::invalid_argument);
}

CUB_TEST("c2h detail uniform offset generator validates destination size", "[c2h][generators]", CUB_SMALL)
{
  REQUIRE_THROWS_AS(
    c2h::detail::gen_uniform_offsets(
      c2h::seed_t{0},
      cuda::std::span<cuda::std::int32_t>{},
      cuda::std::int32_t{1},
      cuda::std::int32_t{0},
      cuda::std::int32_t{1}),
    std::invalid_argument);
}
