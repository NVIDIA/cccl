//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Static error check: a key type whose size is not a power of two is rejected. `char3` is 3 bytes,
// which falls between the supported key widths (1, 2, 4, 8 bytes).

#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

namespace cudax = cuda::experimental;

int main()
{
  using map_t = cudax::cuco::fixed_capacity_map<char3, ::cuda::std::uint8_t>;
  // expected-error {{"key_type size must be a power of two"}}
  static_assert(sizeof(typename map_t::ref_type) > 0);
  return 0;
}
