//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Single umbrella include: must pull in the whole public sharded surface
// (containers, algorithms, place_group) without requiring clients to list
// internal headers.

#include <cuda/experimental/sharded.cuh>

using namespace cuda::experimental::sharded;

int main()
{
  sharded_array<int> empty;
  (void) empty.size();
  (void) empty.is_contiguous();

  shard<int> s;
  (void) s.empty();

  return 0;
}
