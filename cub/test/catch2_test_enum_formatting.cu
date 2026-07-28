// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/cub.cuh>

#include <sstream>
#include <string_view>
#include <version>
#if __cpp_lib_format >= 201907L
#  include <format>
#endif // __cpp_lib_format >= 201907L

#include <c2h/catch2_test_helper.h>

struct OStreamOperatorTester
{
  template <class T>
  void operator()(T value, std::string_view ref) const
  {
    std::ostringstream oss{};
    oss << value;
    REQUIRE(oss.str() == ref);
  }
};

#if __cpp_lib_format >= 201907L
struct FormatTester
{
  template <class T>
  void operator()(T value, std::string_view ref) const
  {
    REQUIRE(std::format("{}", value) == ref);

    // Test that standard specifiers are supported.
    {
      std::string ref_aligned{ref};
      ref_aligned.append(100, ' ');
      ref_aligned.resize(100);
      REQUIRE(std::format("{:<100}", value) == ref_aligned);
    }
  }
};
#endif // __cpp_lib_format >= 201907L

template <class Tester>
void do_test(const Tester& tester)
{
  // BlockHistogramMemoryPreference
  {
    tester(cub::BlockHistogramMemoryPreference::GMEM, "GMEM");
    tester(cub::BlockHistogramMemoryPreference::SMEM, "SMEM");
    tester(cub::BlockHistogramMemoryPreference::BLEND, "BLEND");

    tester(cub::BlockHistogramMemoryPreference(100), "<unknown BlockHistogramMemoryPreference>");
  }

  // RadixSortStoreAlgorithm
  {
    tester(cub::RadixSortStoreAlgorithm::RADIX_SORT_STORE_DIRECT, "RADIX_SORT_STORE_DIRECT");
    tester(cub::RadixSortStoreAlgorithm::RADIX_SORT_STORE_ALIGNED, "RADIX_SORT_STORE_ALIGNED");

    tester(cub::RadixSortStoreAlgorithm(100), "<unknown RadixSortStoreAlgorithm>");
  }

  // BlockLoadAlgorithm
  {
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_DIRECT, "BLOCK_LOAD_DIRECT");
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_STRIPED, "BLOCK_LOAD_STRIPED");
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE, "BLOCK_LOAD_VECTORIZE");
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_TRANSPOSE, "BLOCK_LOAD_TRANSPOSE");
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE, "BLOCK_LOAD_WARP_TRANSPOSE");
    tester(cub::BlockLoadAlgorithm::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED, "BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED");

    tester(cub::BlockLoadAlgorithm(100), "<unknown BlockLoadAlgorithm>");
  }

  // RadixRankAlgorithm
  {
    tester(cub::RadixRankAlgorithm::RADIX_RANK_BASIC, "RADIX_RANK_BASIC");
    tester(cub::RadixRankAlgorithm::RADIX_RANK_MEMOIZE, "RADIX_RANK_MEMOIZE");
    tester(cub::RadixRankAlgorithm::RADIX_RANK_MATCH, "RADIX_RANK_MATCH");
    tester(cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ANY, "RADIX_RANK_MATCH_EARLY_COUNTS_ANY");
    tester(cub::RadixRankAlgorithm::RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR, "RADIX_RANK_MATCH_EARLY_COUNTS_ATOMIC_OR");

    tester(cub::RadixRankAlgorithm(100), "<unknown RadixRankAlgorithm>");
  }

  // BlockReduceAlgorithm
  {
    tester(cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY");
    tester(cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING, "BLOCK_REDUCE_RAKING");
    tester(cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS, "BLOCK_REDUCE_WARP_REDUCTIONS");
    tester(cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
           "BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC");

    tester(cub::BlockReduceAlgorithm(100), "<unknown BlockReduceAlgorithm>");
  }

  // BlockScanAlgorithm
  {
    tester(cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING, "BLOCK_SCAN_RAKING");
    tester(cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE, "BLOCK_SCAN_RAKING_MEMOIZE");
    tester(cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS, "BLOCK_SCAN_WARP_SCANS");

    tester(cub::BlockScanAlgorithm(100), "<unknown BlockScanAlgorithm>");
  }

  // BlockStoreAlgorithm
  {
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_DIRECT, "BLOCK_STORE_DIRECT");
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_STRIPED, "BLOCK_STORE_STRIPED");
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE, "BLOCK_STORE_VECTORIZE");
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_TRANSPOSE, "BLOCK_STORE_TRANSPOSE");
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE, "BLOCK_STORE_WARP_TRANSPOSE");
    tester(cub::BlockStoreAlgorithm::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED, "BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED");

    tester(cub::BlockStoreAlgorithm(100), "<unknown BlockStoreAlgorithm>");
  }

  // LookbackDelayAlgorithm
  {
    tester(cub::LookbackDelayAlgorithm::no_delay, "LookbackDelayAlgorithm::no_delay");
    tester(cub::LookbackDelayAlgorithm::fixed_delay, "LookbackDelayAlgorithm::fixed_delay");
    tester(cub::LookbackDelayAlgorithm::exponential_backoff, "LookbackDelayAlgorithm::exponential_backoff");
    tester(cub::LookbackDelayAlgorithm::exponential_backoff_jitter,
           "LookbackDelayAlgorithm::exponential_backoff_jitter");
    tester(cub::LookbackDelayAlgorithm::exponential_backoff_jitter_window,
           "LookbackDelayAlgorithm::exponential_backoff_jitter_window");
    tester(cub::LookbackDelayAlgorithm::exponential_backon_jitter_window,
           "LookbackDelayAlgorithm::exponential_backon_jitter_window");
    tester(cub::LookbackDelayAlgorithm::exponential_backon_jitter, "LookbackDelayAlgorithm::exponential_backon_jitter");
    tester(cub::LookbackDelayAlgorithm::exponential_backon, "LookbackDelayAlgorithm::exponential_backon");
    tester(cub::LookbackDelayAlgorithm::__reduce_by_key, "LookbackDelayAlgorithm::__reduce_by_key");

    tester(cub::LookbackDelayAlgorithm(100), "<unknown LookbackDelayAlgorithm>");
  }

  // RadixSortAlgorithm
  {
    tester(cub::RadixSortAlgorithm::multi_pass, "RadixSortAlgorithm::multi_pass");
    tester(cub::RadixSortAlgorithm::onesweep, "RadixSortAlgorithm::onesweep");

    tester(cub::RadixSortAlgorithm(100), "<unknown RadixSortAlgorithm>");
  }

  // ScanAlgorithm
  {
    tester(cub::ScanAlgorithm::lookback, "ScanAlgorithm::lookback");
    tester(cub::ScanAlgorithm::lookahead, "ScanAlgorithm::lookahead");

    tester(cub::ScanAlgorithm(100), "<unknown ScanAlgorithm>");
  }

  // TransformAlgorithm
  {
    tester(cub::TransformAlgorithm::prefetch, "TransformAlgorithm::prefetch");
    tester(cub::TransformAlgorithm::vectorized, "TransformAlgorithm::vectorized");
    tester(cub::TransformAlgorithm::ldgsts, "TransformAlgorithm::ldgsts");
    tester(cub::TransformAlgorithm::ublkcp, "TransformAlgorithm::ublkcp");

    tester(cub::TransformAlgorithm(100), "<unknown TransformAlgorithm>");
  }

  // CacheLoadModifier
  {
    tester(cub::CacheLoadModifier::LOAD_DEFAULT, "LOAD_DEFAULT");
    tester(cub::CacheLoadModifier::LOAD_CA, "LOAD_CA");
    tester(cub::CacheLoadModifier::LOAD_CG, "LOAD_CG");
    tester(cub::CacheLoadModifier::LOAD_CS, "LOAD_CS");
    tester(cub::CacheLoadModifier::LOAD_CV, "LOAD_CV");
    tester(cub::CacheLoadModifier::LOAD_LDG, "LOAD_LDG");
    tester(cub::CacheLoadModifier::LOAD_VOLATILE, "LOAD_VOLATILE");

    tester(cub::CacheLoadModifier(100), "<unknown CacheLoadModifier>");
  }

  // WarpLoadAlgorithm
  {
    tester(cub::WarpLoadAlgorithm::WARP_LOAD_DIRECT, "WARP_LOAD_DIRECT");
    tester(cub::WarpLoadAlgorithm::WARP_LOAD_STRIPED, "WARP_LOAD_STRIPED");
    tester(cub::WarpLoadAlgorithm::WARP_LOAD_VECTORIZE, "WARP_LOAD_VECTORIZE");
    tester(cub::WarpLoadAlgorithm::WARP_LOAD_TRANSPOSE, "WARP_LOAD_TRANSPOSE");

    tester(cub::WarpLoadAlgorithm(100), "<unknown WarpLoadAlgorithm>");
  }

  // WarpStoreAlgorithm
  {
    tester(cub::WarpStoreAlgorithm::WARP_STORE_DIRECT, "WARP_STORE_DIRECT");
    tester(cub::WarpStoreAlgorithm::WARP_STORE_STRIPED, "WARP_STORE_STRIPED");
    tester(cub::WarpStoreAlgorithm::WARP_STORE_VECTORIZE, "WARP_STORE_VECTORIZE");
    tester(cub::WarpStoreAlgorithm::WARP_STORE_TRANSPOSE, "WARP_STORE_TRANSPOSE");

    tester(cub::WarpStoreAlgorithm(100), "<unknown WarpStoreAlgorithm>");
  }
}

C2H_TEST("Enum formatting", "")
{
  do_test(OStreamOperatorTester{});
#if __cpp_lib_format >= 201907L
  do_test(FormatTester{});
#endif // __cpp_lib_format >= 201907L
}
