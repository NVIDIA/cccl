//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/std/cstdint>

#include <vector>

#include <nvbench/nvbench.cuh>
#include <nvbench/range.cuh>

namespace cuda::experimental::cuco::benchmark::defaults
{
//! Key types covered by the default CUCO benchmark type axes.
using key_type_range = ::nvbench::type_list<::nvbench::int32_t, ::nvbench::int64_t>;
//! Value types covered by the default CUCO benchmark type axes.
using value_type_range = ::nvbench::type_list<::nvbench::int32_t, ::nvbench::int64_t>;

//! Default number of inputs used when sweeping another benchmark axis.
inline constexpr auto n = ::nvbench::int64_t{100'000'000};
//! Default fixed-capacity map target occupancy.
inline constexpr auto occupancy = 0.5;
//! Default lookup matching rate for contains-style benchmarks.
inline constexpr auto matching_rate = 1.0;
//! Default deterministic seed used by benchmark data generators.
inline constexpr auto seed = ::cuda::std::uint32_t{42};

//! Input-size sweep that remains cacheable for direct comparisons with CUCO benchmarks.
inline const auto n_range_cache = ::std::vector<::nvbench::int64_t>{8'000, 80'000, 800'000, 8'000'000, 80'000'000};
//! Occupancy sweep used by fixed-capacity container benchmarks.
inline const auto occupancy_range = ::nvbench::range(0.1, 0.9, 0.1);
//! Average multiplicity sweep for duplicate-key distributions.
inline const auto multiplicity_range = ::std::vector<double>{1.0, 2.0, 4.0, 8.0, 16.0};
//! Matching-rate sweep used by contains-style benchmarks.
inline const auto matching_rate_range = ::nvbench::range(0.1, 1.0, 0.1);
} // namespace cuda::experimental::cuco::benchmark::defaults
