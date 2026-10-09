//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_SCAN_COMMON_H
#define CUDAX_TEST_MULTI_GPU_SCAN_COMMON_H

#include <cuda/std/cstddef>

#include <numeric>
#include <vector>

namespace scan_test_util
{
// A scan is global: rank `r` sees the prefix of every preceding rank. So the reference scans the
// concatenation of all ranks and then hands back the slice that rank `r` owns. `scan_fn` performs
// the host-side scan, which is the only part that differs between `inclusive_scan` and
// `exclusive_scan`.
template <class T, class ScanFn>
[[nodiscard]] std::vector<T>
expected_for_rank(int rank, const std::vector<std::vector<T>>& inputs_by_rank, ScanFn scan_fn)
{
  std::vector<T> reference;

  for (const auto& values : inputs_by_rank)
  {
    reference.insert(reference.end(), values.begin(), values.end());
  }

  std::vector<T> scan(reference.size());
  scan_fn(reference.begin(), reference.end(), scan.begin());

  cuda::std::size_t offset = 0;
  for (int r = 0; r < rank; ++r)
  {
    offset += inputs_by_rank[static_cast<cuda::std::size_t>(r)].size();
  }

  const auto count = inputs_by_rank[static_cast<cuda::std::size_t>(rank)].size();
  return {scan.begin() + offset, scan.begin() + offset + count};
}

// `std::inclusive_scan` takes the operator before the init, `std::exclusive_scan` after it, so
// each algorithm gets its own wrapper around the shared slicing above.
template <class T, class Op>
[[nodiscard]] std::vector<T>
inclusive_expected_for_rank(int rank, const std::vector<std::vector<T>>& inputs_by_rank, const T& init, Op op)
{
  return expected_for_rank(rank, inputs_by_rank, [&](auto first, auto last, auto out) {
    std::inclusive_scan(first, last, out, op, init);
  });
}

template <class T, class Op>
[[nodiscard]] std::vector<T>
exclusive_expected_for_rank(int rank, const std::vector<std::vector<T>>& inputs_by_rank, const T& init, Op op)
{
  return expected_for_rank(rank, inputs_by_rank, [&](auto first, auto last, auto out) {
    std::exclusive_scan(first, last, out, init, op);
  });
}
} // namespace scan_test_util

#endif // CUDAX_TEST_MULTI_GPU_SCAN_COMMON_H
