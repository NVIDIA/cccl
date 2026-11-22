// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off
// CLANG_TIDY_CONFIG_BEGIN
// Checks: cccl-prefer-addressof
// CheckOptions: cccl-prefer-addressof.IgnoreTestFiles=false
// CLANG_TIDY_CONFIG_END

#include <cuda/std/memory>

// Exercise the check inside its supported namespace.
namespace cuda::std::test
{

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

template <int N, class T>
struct dependent_members
{
  struct storage
  {
    int counter;
    int offsets[N][4];
    T value;
  };

  storage& s;

  int* scalar_address()
  {
    return &s.counter;
  }

  int* array_element_address(int row)
  {
    return &s.offsets[row][0];
  }

  T* dependent_address()
  {
    return &s.value;
    // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
    // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
    // CHECK-FIXES-NOT: #include
    // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
    // CHECK-FIXES: Length: 0
    // CHECK-FIXES-NEXT: ReplacementText: "#include <cuda/std/__memory/addressof.h>\n\n"
    // CHECK-FIXES: Length: 8
    // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(s.value){{'?}}{{$}}
  }
};

void instantiate(dependent_members<2, int>& object)
{
  (void) object.scalar_address();
  (void) object.array_element_address(0);
  (void) object.dependent_address();
}

// CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}

// Reduced from cub/cub/block/block_radix_rank.cuh: BlockRadixRank.
template <int BlockThreads, int CounterLanes, int PackingRatio>
struct radix_rank
{
  using DigitCounter = unsigned short;

  struct TempStorage
  {
    union Aliasable
    {
      DigitCounter digit_counters[CounterLanes + 1][BlockThreads][PackingRatio];
      unsigned int raking_grid[BlockThreads][CounterLanes + 1];
    } aliasable;
  };

  TempStorage& temp_storage;

  template <int KeysPerThread>
  void rank_keys(int counter_lane, int linear_tid, int sub_counter)
  {
    DigitCounter* digit_counters[KeysPerThread];
    digit_counters[0] = &temp_storage.aliasable.digit_counters[counter_lane][linear_tid][sub_counter];
    (void) digit_counters;
  }
};

// Reduced from cub/cub/block/block_radix_rank.cuh: BlockRadixRankMatchEarlyCounts.
template <int BlockThreads, int RadixBits, int NumParts>
struct radix_rank_match
{
  static constexpr int BLOCK_WARPS = BlockThreads / 32;
  static constexpr int RADIX_DIGITS = 1 << RadixBits;

  struct TempStorage
  {
    union
    {
      int warp_offsets[BLOCK_WARPS][RADIX_DIGITS];
      int warp_histograms[BLOCK_WARPS][RADIX_DIGITS][NumParts];
    };
    unsigned int match_masks[BLOCK_WARPS][RADIX_DIGITS];
  };

  template <int KeysPerThread>
  struct RankInternal
  {
    TempStorage& s;

    void rank_keys(int warp, int bin, int part)
    {
      unsigned int* match_masks = &s.match_masks[warp][0];
      int* warp_offsets = &s.warp_offsets[warp][0];
      int* histogram = &s.warp_histograms[warp][bin][part];
      (void) match_masks;
      (void) warp_offsets;
      (void) histogram;
    }
  };
};

void instantiate_2(radix_rank<64, 16, 2>& rank, radix_rank_match<64, 4, 2>::RankInternal<4>& match)
{
  rank.rank_keys<4>(0, 0, 0);
  match.rank_keys(0, 0, 0);
}

template <class T>
T* dependent_address(T& value)
{
  return &value;
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: do not use bare '&' on type-dependent decl-ref expressions, use cuda::std::addressof() instead. [cccl-prefer-addressof]
  // CHECK-MESSAGES-NOT: {{:[0-9]+:[0-9]+: warning:}}
  // CHECK-FIXES-NOT: #include
  // CHECK-FIXES-LABEL: - DiagnosticName: cccl-prefer-addressof
  // CHECK-FIXES: Length: 6
  // CHECK-FIXES-NEXT: ReplacementText: {{'?}}::cuda::std::addressof(value){{'?}}{{$}}
}

// CHECK-FIXES-NOT: #include
// CHECK-FIXES-NOT: DiagnosticName:
} // namespace cuda::std::test

// clang-format on
