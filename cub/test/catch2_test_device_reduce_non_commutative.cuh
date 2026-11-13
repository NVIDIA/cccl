// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cuda/std/cstdint>

#include <ostream>

// A run [first, last] of input indices. Two runs concatenate exactly only if the second starts right after the first,
// so the reduced run shows whether every item was combined once, with its neighbors, in input order.
struct run_t
{
  cuda::std::int64_t first;
  cuda::std::int64_t last;
  bool in_order;

  __host__ __device__ friend bool operator==(const run_t& lhs, const run_t& rhs)
  {
    return lhs.first == rhs.first && lhs.last == rhs.last && lhs.in_order == rhs.in_order;
  }

  friend std::ostream& operator<<(std::ostream& os, const run_t& run)
  {
    return os << '{' << run.first << ", " << run.last << ", " << (run.in_order ? "in order" : "out of order") << '}';
  }
};

struct concatenate_runs_t
{
  __host__ __device__ run_t operator()(const run_t& lhs, const run_t& rhs) const
  {
    return {lhs.first, rhs.last, lhs.in_order && rhs.in_order && lhs.last + 1 == rhs.first};
  }
};

struct index_to_run_t
{
  __host__ __device__ run_t operator()(cuda::std::int64_t index) const
  {
    return {index, index, true};
  }
};

// The initial value is the run that ends right before the first item
inline constexpr run_t initial_run{-1, -1, true};

[[nodiscard]] inline run_t expected_run(cuda::std::int64_t num_items)
{
  return {-1, num_items - 1, true};
}
