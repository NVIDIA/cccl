// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_radix_rank.cuh>

// Verifies that BlockRadixRankMatchEarlyCounts rejects ranking in multiple parts if the digit count is not a multiple
// of the warp size. The check fires as soon as the class is instantiated.

// 2^4 = 16 digits are not a multiple of the warp size (32)
using rank_t =
  cub::BlockRadixRankMatchEarlyCounts<128, 4, false, cub::BLOCK_SCAN_WARP_SCANS, cub::WARP_MATCH_ANY, /* NumParts */ 2>;

int main()
{
  // expected-error {{"RadixBits must be a multiple of the warp size"}}
  return static_cast<int>(sizeof(rank_t));
}
