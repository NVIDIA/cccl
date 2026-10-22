//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Static error check: a slot (key/payload pair) whose size is not a power of two is rejected. A
// 1-byte key with a 5-byte payload forms a 6-byte slot, which the packed atomic update cannot
// address even though the key size alone is a valid power of two.

#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

namespace cudax = cuda::experimental;

struct byte5
{
  unsigned char a, b, c, d, e;
};

int main()
{
  using map_t = cudax::cuco::fixed_capacity_map<::cuda::std::uint8_t, byte5>;
  // expected-error {{"value_type size must be a power of two"}}
  static_assert(sizeof(typename map_t::ref_type) > 0);
  return 0;
}
