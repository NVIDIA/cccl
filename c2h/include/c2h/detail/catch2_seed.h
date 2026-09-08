// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/limits>

#include <cstddef>

#include <c2h/detail/env.cuh>
#include <c2h/generator_types.h>
#include <catch2/generators/catch_generators_all.hpp>

namespace c2h
{
inline std::size_t get_override_seed_count()
{
  static const std::size_t override_seeds = [] {
    // Setting this environment variable forces a fixed number of seeds to be generated, regardless of the requested
    // count. Set to 1 to reduce redundant, expensive testing when using sanitizers, etc.
    const auto parsed = ::c2h::detail::get_env_as_long_long("C2H_SEED_COUNT_OVERRIDE");
    return parsed > 0 ? static_cast<std::size_t>(parsed) : 0;
  }();

  return override_seeds;
}

inline std::size_t adjust_seed_count(std::size_t requested)
{
  static const std::size_t override_seeds = ::c2h::get_override_seed_count();
  return override_seeds != 0 ? override_seeds : requested;
}
} // namespace c2h

#define C2H_SEED(N)                                                                         \
  c2h::seed_t                                                                               \
  {                                                                                         \
    GENERATE_COPY(take(c2h::adjust_seed_count(N),                                           \
                       random(::cuda::std::numeric_limits<unsigned long long int>::min(),   \
                              ::cuda::std::numeric_limits<unsigned long long int>::max()))) \
  }
