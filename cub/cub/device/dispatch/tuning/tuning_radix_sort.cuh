// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_radix_sort_downsweep.cuh>
#include <cub/agent/agent_radix_sort_histogram.cuh>
#include <cub/agent/agent_radix_sort_onesweep.cuh>
#include <cub/agent/agent_radix_sort_upsweep.cuh>
#include <cub/agent/agent_scan.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
#include <cub/util_device.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

//! The algorithm to use for radix sorting.
enum class RadixSortAlgorithm
{
  multi_pass, //!< Multi-pass radix sort (upsweep + scan + downsweep per digit)
  onesweep //!< Single-pass radix sort using decoupled look-back
};

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, RadixSortAlgorithm algorithm)
{
  switch (algorithm)
  {
    case RadixSortAlgorithm::multi_pass:
      return os << "RadixSortAlgorithm::multi_pass";
    case RadixSortAlgorithm::onesweep:
      return os << "RadixSortAlgorithm::onesweep";
    default:
      return os << "RadixSortAlgorithm::unknown(" << static_cast<int>(algorithm) << ")";
  }
}
#endif // _CCCL_HOSTED()

//! The tuning policy for the histogram pass of @ref DeviceRadixSort (used by the onesweep algorithm).
struct RadixSortHistogramPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread

  //! The number of private histogram partitions in shared memory each histogram is split during counting to reduce the
  //! contention of atomic operations
  int num_private_partitions;
  int radix_bits; //!< Number of bits per radix digit

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortHistogramPolicy& lhs, const RadixSortHistogramPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.num_private_partitions == rhs.num_private_partitions && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortHistogramPolicy& lhs, const RadixSortHistogramPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortHistogramPolicy& p)
  {
    return os
        << "RadixSortHistogramPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .num_private_partitions = " << p.num_private_partitions
        << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for the exclusive sum pass of @ref DeviceRadixSort (used by the onesweep algorithm).
struct RadixSortExclusiveSumPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int radix_bits; //!< Number of bits per radix digit

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortExclusiveSumPolicy& lhs, const RadixSortExclusiveSumPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortExclusiveSumPolicy& lhs, const RadixSortExclusiveSumPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortExclusiveSumPolicy& p)
  {
    return os << "RadixSortExclusiveSumPolicy { .threads_per_block = " << p.threads_per_block
              << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for the onesweep pass of @ref DeviceRadixSort.
struct RadixSortOnesweepPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  RadixSortStoreAlgorithm store_algorithm; //!< The @ref RadixSortStoreAlgorithm used for writing results
  RadixRankAlgorithm rank_algorithm; //!< The @ref RadixRankAlgorithm used for ranking keys
  BlockScanAlgorithm scan_algorithm; //!< The @ref BlockScanAlgorithm used for scanning within a thread block

  //! The number of private histogram partitions in shared memory each histogram is split during the ranking phase to
  //! reduce the contention of atomic operations. Ignored if @p rank_algorithm is not one of
  //! RADIX_RANK_MATCH_EARLY_COUNTS_*
  int rank_num_private_partitions;

  int radix_bits; //!< Number of bits per radix digit

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortOnesweepPolicy& lhs, const RadixSortOnesweepPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.store_algorithm == rhs.store_algorithm && lhs.rank_algorithm == rhs.rank_algorithm
        && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.rank_num_private_partitions == rhs.rank_num_private_partitions && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortOnesweepPolicy& lhs, const RadixSortOnesweepPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortOnesweepPolicy& p)
  {
    return os
        << "RadixSortOnesweepPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .store_algorithm = " << p.store_algorithm
        << ", .rank_algorithm = " << p.rank_algorithm << ", .scan_algorithm = " << p.scan_algorithm
        << ", .rank_num_private_partitions = " << p.rank_num_private_partitions << ", .radix_bits = " << p.radix_bits
        << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for the downsweep pass (and single-tile path) of @ref DeviceRadixSort.
struct RadixSortDownsweepPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading items from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  RadixRankAlgorithm rank_algorithm; //!< The @ref RadixRankAlgorithm used for ranking keys
  BlockScanAlgorithm scan_algorithm; //!< The @ref BlockScanAlgorithm used for scanning within a thread block
  int radix_bits; //!< Number of bits per radix digit

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortDownsweepPolicy& lhs, const RadixSortDownsweepPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.rank_algorithm == rhs.rank_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortDownsweepPolicy& lhs, const RadixSortDownsweepPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortDownsweepPolicy& p)
  {
    return os
        << "RadixSortDownsweepPolicy { .threads_per_block = " << p.threads_per_block
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .rank_algorithm = " << p.rank_algorithm
        << ", .scan_algorithm = " << p.scan_algorithm << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for the upsweep pass of @ref DeviceRadixSort (used by the multi-pass algorithm).
struct RadixSortUpsweepPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  int radix_bits; //!< Number of bits per radix digit

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortUpsweepPolicy& lhs, const RadixSortUpsweepPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.radix_bits == rhs.radix_bits;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortUpsweepPolicy& lhs, const RadixSortUpsweepPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortUpsweepPolicy& p)
  {
    return os
        << "RadixSortUpsweepPolicy { .threads_per_block = " << p.threads_per_block << ", .items_per_thread = "
        << p.items_per_thread << ", .load_modifier = " << p.load_modifier << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all algorithms in @ref DeviceRadixSort.
struct RadixSortPolicy
{
  RadixSortAlgorithm algorithm; //!< The radix sort algorithm to use
  RadixSortHistogramPolicy histogram; //!< Histogram pass policy (onesweep only)
  RadixSortExclusiveSumPolicy exclusive_sum; //!< Exclusive sum pass policy (onesweep only)
  RadixSortOnesweepPolicy onesweep; //!< Onesweep pass policy
  ScanPolicy scan; //!< Scan policy (multi-pass only)
  RadixSortDownsweepPolicy downsweep; //!< Downsweep pass policy (multi-pass only)
  RadixSortDownsweepPolicy alt_downsweep; //!< Alternate downsweep pass policy with fewer radix bits
  RadixSortUpsweepPolicy upsweep; //!< Upsweep pass policy (multi-pass only)
  RadixSortUpsweepPolicy alt_upsweep; //!< Alternate upsweep pass policy with fewer radix bits
  RadixSortDownsweepPolicy single_tile; //!< Single-tile sort policy for small inputs

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RadixSortPolicy& lhs, const RadixSortPolicy& rhs) noexcept
  {
    return lhs.algorithm == rhs.algorithm && lhs.histogram == rhs.histogram && lhs.exclusive_sum == rhs.exclusive_sum
        && lhs.onesweep == rhs.onesweep && lhs.scan == rhs.scan && lhs.downsweep == rhs.downsweep
        && lhs.alt_downsweep == rhs.alt_downsweep && lhs.upsweep == rhs.upsweep && lhs.alt_upsweep == rhs.alt_upsweep
        && lhs.single_tile == rhs.single_tile;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RadixSortPolicy& lhs, const RadixSortPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RadixSortPolicy& p)
  {
    return os
        << "RadixSortPolicy { .algorithm = " << p.algorithm << ", .histogram = " << p.histogram
        << ", .exclusive_sum = " << p.exclusive_sum << ", .onesweep = " << p.onesweep << ", .scan = " << p.scan
        << ", .downsweep = " << p.downsweep << ", .alt_downsweep = " << p.alt_downsweep << ", .upsweep = " << p.upsweep
        << ", .alt_upsweep = " << p.alt_upsweep << ", .single_tile = " << p.single_tile << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::radix_sort
{
using detail::scan::make_mem_scaled_lookback_scan_policy;

_CCCL_HOST_DEVICE_API constexpr auto make_reg_scaled_radix_sort_onesweep_policy(
  int nominal_4b_threads_per_block,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  RadixSortStoreAlgorithm store_algorithm,
  RadixRankAlgorithm rank_algorithm,
  BlockScanAlgorithm scan_algorithm,
  int rank_num_private_partitions,
  int radix_bits) -> RadixSortOnesweepPolicy
{
  const auto scaled = scale_reg_bound(nominal_4b_threads_per_block, nominal_4b_items_per_thread, compute_t_size);
  return RadixSortOnesweepPolicy{
    scaled.threads_per_block,
    scaled.items_per_thread,
    store_algorithm,
    rank_algorithm,
    scan_algorithm,
    rank_num_private_partitions,
    radix_bits};
}

_CCCL_HOST_DEVICE_API constexpr auto make_reg_scaled_radix_sort_downsweep_policy(
  int nominal_4b_threads_per_block,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  BlockLoadAlgorithm load_algorithm,
  CacheLoadModifier load_modifier,
  RadixRankAlgorithm rank_algorithm,
  BlockScanAlgorithm scan_algorithm,
  int radix_bits) -> RadixSortDownsweepPolicy
{
  const auto scaled = scale_reg_bound(nominal_4b_threads_per_block, nominal_4b_items_per_thread, compute_t_size);
  return RadixSortDownsweepPolicy{
    scaled.threads_per_block,
    scaled.items_per_thread,
    load_algorithm,
    load_modifier,
    rank_algorithm,
    scan_algorithm,
    radix_bits};
}

_CCCL_HOST_DEVICE_API constexpr auto make_reg_scaled_radix_sort_upsweep_policy(
  int nominal_4b_threads_per_block,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  CacheLoadModifier load_modifier,
  int radix_bits) -> RadixSortUpsweepPolicy
{
  const auto scaled = scale_reg_bound(nominal_4b_threads_per_block, nominal_4b_items_per_thread, compute_t_size);
  return RadixSortUpsweepPolicy{scaled.threads_per_block, scaled.items_per_thread, load_modifier, radix_bits};
}

// TODO(bgruber): remove for CCCL 4.0 when we drop the public radix sort dispatcher
// sm90 default
template <size_t KeySize, size_t ValueSize, size_t OffsetSize>
struct sm90_small_key_tuning
{
  static constexpr int threads = 384;
  static constexpr int items   = 23;
};

// clang-format off

// keys
template <> struct sm90_small_key_tuning<1,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<1,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 4> { static constexpr int threads = 512; static constexpr int items = 19; };
template <> struct sm90_small_key_tuning<2,  0, 8> { static constexpr int threads = 512; static constexpr int items = 19; };

// pairs  8:xx
template <> struct sm90_small_key_tuning<1,  1, 4> { static constexpr int threads = 512; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<1,  1, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<1,  2, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  2, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<1,  4, 8> { static constexpr int threads = 512; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<1,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<1,  8, 8> { static constexpr int threads = 384; static constexpr int items = 18; };
template <> struct sm90_small_key_tuning<1, 16, 4> { static constexpr int threads = 512; static constexpr int items = 22; };
template <> struct sm90_small_key_tuning<1, 16, 8> { static constexpr int threads = 512; static constexpr int items = 22; };

// pairs 16:xx
template <> struct sm90_small_key_tuning<2,  1, 4> { static constexpr int threads = 384; static constexpr int items = 14; };
template <> struct sm90_small_key_tuning<2,  1, 8> { static constexpr int threads = 384; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  2, 4> { static constexpr int threads = 384; static constexpr int items = 15; };
template <> struct sm90_small_key_tuning<2,  2, 8> { static constexpr int threads = 448; static constexpr int items = 16; };
template <> struct sm90_small_key_tuning<2,  4, 4> { static constexpr int threads = 512; static constexpr int items = 17; };
template <> struct sm90_small_key_tuning<2,  4, 8> { static constexpr int threads = 512; static constexpr int items = 12; };
template <> struct sm90_small_key_tuning<2,  8, 4> { static constexpr int threads = 384; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2,  8, 8> { static constexpr int threads = 512; static constexpr int items = 23; };
template <> struct sm90_small_key_tuning<2, 16, 4> { static constexpr int threads = 512; static constexpr int items = 21; };
template <> struct sm90_small_key_tuning<2, 16, 8> { static constexpr int threads = 576; static constexpr int items = 22; };
// clang-format on

// TODO(bgruber): remove for CCCL 4.0 when we drop the public radix sort dispatcher
// sm100 default
template <typename ValueT, size_t KeySize, size_t ValueSize, size_t OffsetSize>
struct sm100_small_key_tuning : sm90_small_key_tuning<KeySize, ValueSize, OffsetSize>
{};

// clang-format off

// keys

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  0, 4> : sm90_small_key_tuning<1, 0, 4> {};

// ipt_20.tpb_512 1.013282  0.967525  1.015764  1.047982
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  0, 4> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_21.tpb_512 1.002873  0.994608  1.004196  1.019301
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  0, 4> { static constexpr int threads = 512; static constexpr int items = 21; };

// ipt_14.tpb_320 1.256020  1.000000  1.228182  1.486711
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  0, 4> { static constexpr int threads = 320; static constexpr int items = 14; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 16,  0, 4> : sm90_small_key_tuning<16, 0, 4> {};

// ipt_20.tpb_512 1.089698  0.979276  1.079822  1.199378
template <> struct sm100_small_key_tuning<float, 4,  0, 4> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_18.tpb_288 1.049258  0.985085  1.042400  1.107771
template <> struct sm100_small_key_tuning<double, 8,  0, 4> { static constexpr int threads = 288; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  0, 8> : sm90_small_key_tuning<1, 0, 8> {};

// ipt_20.tpb_384 1.038445  1.015608  1.037620  1.068105
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  0, 8> { static constexpr int threads = 384; static constexpr int items = 20; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  0, 8> : sm90_small_key_tuning<4, 0, 8> {};

// ipt_18.tpb_320 1.248354  1.000000  1.220666  1.446929
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  0, 8> { static constexpr int threads = 320; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 16,  0, 8> : sm90_small_key_tuning<16, 0, 8> {};

// ipt_20.tpb_512 1.021557  0.981437  1.018920  1.039977
template <> struct sm100_small_key_tuning<float, 4,  0, 8> { static constexpr int threads = 512; static constexpr int items = 20; };

// ipt_21.tpb_256 1.068590  0.986635  1.059704  1.144921
template <> struct sm100_small_key_tuning<double, 8,  0, 8> { static constexpr int threads = 256; static constexpr int items = 21; };

// pairs 1-byte key

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  1, 4> : sm90_small_key_tuning<1, 1, 4> {};

// ipt_18.tpb_512 1.011463  0.978807  1.010106  1.024056
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  2, 4> { static constexpr int threads = 512; static constexpr int items = 18; };

// ipt_18.tpb_512 1.008207  0.980377  1.007132  1.022155
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  4, 4> { static constexpr int threads = 512; static constexpr int items = 18; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  8, 4> { static constexpr int threads = 288; static constexpr int items = 16; };

// ipt_21.tpb_576 1.044274  0.979145  1.038723  1.072068
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  16, 4> { static constexpr int threads = 576; static constexpr int items = 21; };

// ipt_20.tpb_384 1.008881  0.968750  1.006846  1.026910
// todo(@gonidelis): insignificant performance gain, need more runs.
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  1, 8> { static constexpr int threads = 384; static constexpr int items = 20; };

// ipt_22.tpb_256 1.015597  0.966038  1.011167  1.045921
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  2, 8> { static constexpr int threads = 256; static constexpr int items = 22; };

// ipt_15.tpb_384 1.029730  0.972699  1.029066  1.067894
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  4, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  8, 8> { static constexpr int threads = 256; static constexpr int items = 17; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 1,  16, 8> : sm90_small_key_tuning<1, 16, 8> {};


// pairs 2-byte key

// ipt_20.tpb_448  1.031929  0.936849  1.023411  1.075172
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  1, 4> { static constexpr int threads = 448; static constexpr int items = 20; };

// ipt_23.tpb_384 1.104683  0.939335  1.087342  1.234988
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  2, 4> { static constexpr int threads = 384; static constexpr int items = 23; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  4, 4>  : sm90_small_key_tuning<2, 4, 4> {};

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  8, 4> { static constexpr int threads = 256; static constexpr int items = 17; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  16, 4> : sm90_small_key_tuning<2, 16, 4> {};

// ipt_15.tpb_384 1.093598  1.000000  1.088111  1.183369
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  1, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_15.tpb_576 1.040476  1.000333  1.037060  1.084850
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  2, 8> { static constexpr int threads = 576; static constexpr int items = 15; };

// ipt_18.tpb_512 1.096819  0.953488  1.082026  1.209533
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  4, 8> { static constexpr int threads = 512; static constexpr int items = 18; };

// todo(@gonidelis): regresses for large problem sizes.
// template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  8, 8> { static constexpr int threads = 288; static constexpr int items = 16; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 2,  16, 8> : sm90_small_key_tuning<2, 16, 8> {};


// pairs 4-byte key

// ipt_21.tpb_416 1.237956  1.001909  1.210882  1.469981
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  1, 4> { static constexpr int threads = 416; static constexpr int items = 21; };

// ipt_17.tpb_512 1.022121  1.012346  1.022439  1.038524
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  2, 4> { static constexpr int threads = 512; static constexpr int items = 17; };

// ipt_20.tpb_448 1.012688  0.999531  1.011865  1.028513
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  4, 4>  { static constexpr int threads = 448; static constexpr int items = 20; };

// ipt_15.tpb_384 1.006872  0.998651  1.008374  1.026118
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  8, 4> { static constexpr int threads = 384; static constexpr int items = 15; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  16, 4> : sm90_small_key_tuning<4, 16, 4> {};

// ipt_17.tpb_512 1.080000  0.927362  1.066211  1.172959
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  1, 8> { static constexpr int threads = 512; static constexpr int items = 17; };

// ipt_15.tpb_384 1.068529  1.000000  1.062277  1.135281
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  2, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_21.tpb_448  1.080642  0.927713  1.064758  1.191177
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  4, 8> { static constexpr int threads = 448; static constexpr int items = 21; };

// ipt_13.tpb_448 1.019046  0.991228  1.016971  1.039712
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  8, 8> { static constexpr int threads = 448; static constexpr int items = 13; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 4,  16, 8> : sm90_small_key_tuning<4, 16, 8> {};

// pairs 8-byte key

// ipt_17.tpb_256 1.276445  1.025562  1.248511  1.496947
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  1, 4> { static constexpr int threads = 256; static constexpr int items = 17; };

// ipt_12.tpb_352 1.128086  1.040000  1.117960  1.207254
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  2, 4> { static constexpr int threads = 352; static constexpr int items = 12; };

// ipt_12.tpb_352 1.132699  1.040000  1.122676  1.207716
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  4, 4>  { static constexpr int threads = 352; static constexpr int items = 12; };

// ipt_18.tpb_256 1.266745  0.995432  1.237754  1.460538
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  8, 4> { static constexpr int threads = 256; static constexpr int items = 18; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  16, 4> : sm90_small_key_tuning<8, 16, 4> {};

// ipt_15.tpb_384 1.007343  0.997656  1.006929  1.047208
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  1, 8> { static constexpr int threads = 384; static constexpr int items = 15; };

// ipt_14.tpb_256 1.186477  1.012683  1.167150  1.332313
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  2, 8> { static constexpr int threads = 256; static constexpr int items = 14; };

// ipt_21.tpb_256 1.220607  1.000239  1.196400  1.390471
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  4, 8> { static constexpr int threads = 256; static constexpr int items = 21; };

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  8, 8> :  sm90_small_key_tuning<8, 8, 8> {};

// same as previous tuning
template <typename ValueT> struct sm100_small_key_tuning<ValueT, 8,  16, 8> : sm90_small_key_tuning<8, 16, 8> {};
// clang-format on

struct small_key_tuning_values
{
  int threads;
  int items;
};

_CCCL_HOST_DEVICE_API constexpr auto get_sm90_tuning(int key_size, int value_size, int offset_size)
  -> small_key_tuning_values
{
  // keys
  if (value_size == 0)
  {
    // clang-format off
    if (key_size == 1 && offset_size == 4) return {512,19};
    if (key_size == 1 && offset_size == 8) return {512,19};
    if (key_size == 2 && offset_size == 4) return {512,19};
    if (key_size == 2 && offset_size == 8) return {512,19};
    // clang-format on
  }

  // pairs  8:xx
  if (key_size == 1)
  {
    // clang-format off
    if (value_size ==  1 && offset_size == 4) return {512, 15};
    if (value_size ==  1 && offset_size == 8) return {448, 16};
    if (value_size ==  2 && offset_size == 4) return {512, 17};
    if (value_size ==  2 && offset_size == 8) return {512, 14};
    if (value_size ==  4 && offset_size == 4) return {512, 17};
    if (value_size ==  4 && offset_size == 8) return {512, 14};
    if (value_size ==  8 && offset_size == 4) return {384, 23};
    if (value_size ==  8 && offset_size == 8) return {384, 18};
    if (value_size == 16 && offset_size == 4) return {512, 22};
    if (value_size == 16 && offset_size == 8) return {512, 22};
    // clang-format on
  }

  // pairs  16:xx
  if (key_size == 2)
  {
    // clang-format off
    if (value_size ==  1 && offset_size == 4) return {384, 14};
    if (value_size ==  1 && offset_size == 8) return {384, 16};
    if (value_size ==  2 && offset_size == 4) return {384, 15};
    if (value_size ==  2 && offset_size == 8) return {448, 16};
    if (value_size ==  4 && offset_size == 4) return {512, 17};
    if (value_size ==  4 && offset_size == 8) return {512, 12};
    if (value_size ==  8 && offset_size == 4) return {384, 23};
    if (value_size ==  8 && offset_size == 8) return {512, 23};
    if (value_size == 16 && offset_size == 4) return {512, 21};
    if (value_size == 16 && offset_size == 8) return {576, 22};
    // clang-format on
  }

  return {384, 23};
}

_CCCL_HOST_DEVICE_API constexpr auto get_sm100_tuning(int key_size, int value_size, int offset_size, type_t key_type)
  -> small_key_tuning_values
{
  // keys
  if (value_size == 0)
  {
    if (offset_size == 4)
    {
      // clang-format off

      // if (key_size == 1) // same as previous tuning

      // // ipt_20.tpb_512 1.013282  0.967525  1.015764  1.047982
      // todo(@gonidelis): insignificant performance gain, need more runs.
      if (key_size == 2) return small_key_tuning_values{512,20};

      // ipt_20.tpb_512 1.089698  0.979276  1.079822  1.199378
      if (key_size == 4 && key_type == type_t::float32) return small_key_tuning_values{512,20};

      // ipt_21.tpb_512 1.002873  0.994608  1.004196  1.019301
      // todo(@gonidelis): insignificant performance gain, need more runs.
      if (key_size == 4) return small_key_tuning_values{512,21};

      // ipt_18.tpb_288 1.049258  0.985085  1.042400  1.107771
      if (key_size == 8 && key_type ==  type_t::float64) return small_key_tuning_values{288,18};

      // ipt_14.tpb_320 1.256020  1.000000  1.228182  1.486711
      if (key_size == 8) return small_key_tuning_values{320,14};

      // if (key_size == 16) // same as previous tuning

      // clang-format on
    }
    else if (offset_size == 8)
    {
      // clang-format off

      // if (key_size == 1) // same as previous tuning

      // ipt_20.tpb_384 1.038445  1.015608  1.037620  1.068105
      if (key_size == 2) return small_key_tuning_values{384,20};

      // ipt_20.tpb_512 1.021557  0.981437  1.018920  1.039977
      if (key_size == 4 && key_type == type_t::float32) return small_key_tuning_values{512,20};

      // if (key_size == 4) // same as previous tuning

      // ipt_21.tpb_256 1.068590  0.986635  1.059704  1.144921
      if (key_size == 8 && key_type == type_t::float64) return small_key_tuning_values{256,21};

      // ipt_18.tpb_320 1.248354  1.000000  1.220666  1.446929
      if (key_size == 8) return small_key_tuning_values{320,18};

      // if (key_size == 16) // same as previous tuning

      // clang-format on
    }
  }

  // pairs 1-byte key
  if (key_size == 1)
  {
    // clang-format off

    // if (value_size == 1 && offset_size == 4) // same as previous tuning

    // ipt_18.tpb_512 1.011463  0.978807  1.010106  1.024056
    // todo(@gonidelis): insignificant performance gain, need more runs.
    if (value_size == 2 && offset_size == 4) return small_key_tuning_values{512,18};

    // ipt_18.tpb_512 1.008207  0.980377  1.007132  1.022155
    // todo(@gonidelis): insignificant performance gain, need more runs.
    if (value_size == 4 && offset_size == 4) return small_key_tuning_values{512,18};

    // todo(@gonidelis): regresses for large problem sizes.
    // if (value_size == 8 && offset_size == 4) return small_key_tuning_values{288,16};

    // ipt_21.tpb_576 1.044274  0.979145  1.038723  1.072068
    // todo(@gonidelis): insignificant performance gain, need more runs.
    if (value_size == 16 && offset_size == 4) return small_key_tuning_values{576,21};

    // ipt_20.tpb_384 1.008881  0.968750  1.006846  1.026910
    // todo(@gonidelis): insignificant performance gain, need more runs.
    if (value_size == 1 && offset_size == 8) return small_key_tuning_values{384,20};

    // ipt_22.tpb_256 1.015597  0.966038  1.011167  1.045921
    if (value_size == 2 && offset_size == 8) return small_key_tuning_values{256,22};

    // ipt_15.tpb_384 1.029730  0.972699  1.029066  1.067894
    if (value_size == 4 && offset_size == 8) return small_key_tuning_values{384,15};

    // todo(@gonidelis): regresses for large problem sizes.
    // if (value_size == 8 && offset_size == 8) return small_key_tuning_values{256,17};

    // if (value_size == 16 && offset_size == 8) // same as previous tuning

    // clang-format on
  }

  // pairs 2-byte key
  if (key_size == 2)
  {
    // clang-format off

    // ipt_20.tpb_448  1.031929  0.936849  1.023411  1.075172
    if (value_size == 1 && offset_size == 4) return small_key_tuning_values{448,20};

    // ipt_23.tpb_384 1.104683  0.939335  1.087342  1.234988
    if (value_size == 2 && offset_size == 4) return small_key_tuning_values{384,23};

    // if (value_size == 4 && offset_size == 4) // same as previous tuning

    // todo(@gonidelis): regresses for large problem sizes.
    // if (value_size == 8 && offset_size == 4) return small_key_tuning_values{256, 17};

    // if (value_size == 16 && offset_size == 4) // same as previous tuning

    // ipt_15.tpb_384 1.093598  1.000000  1.088111  1.183369
    if (value_size == 1 && offset_size == 8) return small_key_tuning_values{384, 15};

    // ipt_15.tpb_576 1.040476  1.000333  1.037060  1.084850
    if (value_size == 2 && offset_size == 8) return small_key_tuning_values{576, 15};

    // ipt_18.tpb_512 1.096819  0.953488  1.082026  1.209533
    if (value_size == 4 && offset_size == 8) return small_key_tuning_values{512, 18};

    // todo(@gonidelis): regresses for large problem sizes.
    // if (value_size == 8 && offset_size == 8) return small_key_tuning_values{288, 16};

    // if (value_size == 16 && offset_size == 8) // same as previous tuning

    // clang-format on
  }

  // pairs 4-byte key
  if (key_size == 4)
  {
    // clang-format off

    // ipt_21.tpb_416 1.237956  1.001909  1.210882  1.469981
    if (value_size == 1 && offset_size == 4) return small_key_tuning_values{416,21};

    // ipt_17.tpb_512 1.022121  1.012346  1.022439  1.038524
    if (value_size == 2 && offset_size == 4) return small_key_tuning_values{512,17};

    // ipt_20.tpb_448 1.012688  0.999531  1.011865  1.028513
    if (value_size == 4 && offset_size == 4) return small_key_tuning_values{448,20};

    // ipt_15.tpb_384 1.006872  0.998651  1.008374  1.026118
    if (value_size == 8 && offset_size == 4) return small_key_tuning_values{384,15};

    // if (value_size == 16 && offset_size == 4) // same as previous tuning

    // ipt_17.tpb_512 1.080000  0.927362  1.066211  1.172959
    if (value_size == 1 && offset_size == 8) return small_key_tuning_values{512,17};

    // ipt_15.tpb_384 1.068529  1.000000  1.062277  1.135281
    if (value_size == 2 && offset_size == 8) return small_key_tuning_values{384,15};

    // ipt_21.tpb_448  1.080642  0.927713  1.064758  1.191177
    if (value_size == 4 && offset_size == 8) return small_key_tuning_values{448,21};

    // ipt_13.tpb_448 1.019046  0.991228  1.016971  1.039712
    if (value_size == 8 && offset_size == 8) return small_key_tuning_values{448,13};

    // if (value_size == 16 && offset_size == 8) // same as previous tuning

    // clang-format on
  }

  // pairs 8-byte key
  if (key_size == 8)
  {
    // clang-format off

    // ipt_17.tpb_256 1.276445  1.025562  1.248511  1.496947
    if (value_size == 1 && offset_size == 4) return small_key_tuning_values{256, 17};

    // ipt_12.tpb_352 1.128086  1.040000  1.117960  1.207254
    if (value_size == 2 && offset_size == 4) return small_key_tuning_values{352, 12};

    // ipt_12.tpb_352 1.132699  1.040000  1.122676  1.207716
    if (value_size == 4 && offset_size == 4) return small_key_tuning_values{352, 12};

    // ipt_18.tpb_256 1.266745  0.995432  1.237754  1.460538
    if (value_size == 8 && offset_size == 4) return small_key_tuning_values{256, 18};

    // if (value_size == 16 && offset_size == 4) // same as previous tuning

    // ipt_15.tpb_384 1.007343  0.997656  1.006929  1.047208
    if (value_size == 1 && offset_size == 8) return small_key_tuning_values{384, 15};

    // ipt_14.tpb_256 1.186477  1.012683  1.167150  1.332313
    if (value_size == 2 && offset_size == 8) return small_key_tuning_values{256, 14};

    // ipt_21.tpb_256 1.220607  1.000239  1.196400  1.390471
    if (value_size == 4 && offset_size == 8) return small_key_tuning_values{256, 21};

    // if (value_size == 8 && offset_size == 8) // same as previous tuning

    // if (value_size == 16 && offset_size == 8) // same as previous tuning

    // clang-format on
  }

  return get_sm90_tuning(key_size, value_size, offset_size);
}

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename PolicyT, typename = void>
struct RadixSortPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE RadixSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename StaticPolicyT>
struct RadixSortPolicyWrapper<
  StaticPolicyT,
  ::cuda::std::void_t<typename StaticPolicyT::SingleTilePolicy,
                      typename StaticPolicyT::OnesweepPolicy,
                      typename StaticPolicyT::UpsweepPolicy,
                      typename StaticPolicyT::AltUpsweepPolicy,
                      typename StaticPolicyT::DownsweepPolicy,
                      typename StaticPolicyT::AltDownsweepPolicy,
                      typename StaticPolicyT::HistogramPolicy,
                      typename StaticPolicyT::ScanPolicy,
                      typename StaticPolicyT::ExclusiveSumPolicy,
                      typename StaticPolicyT::SegmentedPolicy,
                      typename StaticPolicyT::AltSegmentedPolicy>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE RadixSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr bool IsOnesweep()
  {
    return StaticPolicyT::ONESWEEP;
  }

  template <typename PolicyT>
  _CCCL_HOST_DEVICE static constexpr int RadixBits(PolicyT /*policy*/)
  {
    return PolicyT::RADIX_BITS;
  }

  template <typename PolicyT>
  _CCCL_HOST_DEVICE static constexpr int ThreadsPerBlock(PolicyT /*policy*/)
  {
    return PolicyT::BLOCK_THREADS;
  }

  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile);
  CUB_DEFINE_SUB_POLICY_GETTER(Onesweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Upsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(AltUpsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Downsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(AltDownsweep);
  CUB_DEFINE_SUB_POLICY_GETTER(Histogram);
  CUB_DEFINE_SUB_POLICY_GETTER(Scan);
  CUB_DEFINE_SUB_POLICY_GETTER(ExclusiveSum);
  CUB_DEFINE_SUB_POLICY_GETTER(Segmented);
  CUB_DEFINE_SUB_POLICY_GETTER(AltSegmented);
};

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename PolicyT>
_CCCL_HOST_DEVICE RadixSortPolicyWrapper<PolicyT> MakeRadixSortPolicyWrapper(PolicyT policy)
{
  return RadixSortPolicyWrapper<PolicyT>{policy};
}

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename DownsweepPolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_downsweep_policy(DownsweepPolicy)
{
  return RadixSortDownsweepPolicy{
    DownsweepPolicy::BLOCK_THREADS,
    DownsweepPolicy::ITEMS_PER_THREAD,
    DownsweepPolicy::LOAD_ALGORITHM,
    DownsweepPolicy::LOAD_MODIFIER,
    DownsweepPolicy::RANK_ALGORITHM,
    DownsweepPolicy::SCAN_ALGORITHM,
    DownsweepPolicy::RADIX_BITS};
};

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename LegacyActivePolicy>
_CCCL_HOST_DEVICE_API constexpr auto convert_policy() -> RadixSortPolicy
{
  using active_policy = LegacyActivePolicy;

  using hist_pol       = typename active_policy::HistogramPolicy;
  const auto histogram = RadixSortHistogramPolicy{
    hist_pol::BLOCK_THREADS, hist_pol::ITEMS_PER_THREAD, hist_pol::NUM_PARTS, hist_pol::RADIX_BITS};

  using exc_sum_pol        = typename active_policy::ExclusiveSumPolicy;
  const auto exclusive_sum = RadixSortExclusiveSumPolicy{exc_sum_pol::BLOCK_THREADS, exc_sum_pol::RADIX_BITS};

  using one_pol       = typename active_policy::OnesweepPolicy;
  const auto onesweep = RadixSortOnesweepPolicy{
    one_pol::BLOCK_THREADS,
    one_pol::ITEMS_PER_THREAD,
    one_pol::STORE_ALGORITHM,
    one_pol::RANK_ALGORITHM,
    one_pol::SCAN_ALGORITHM,
    one_pol::RANK_NUM_PARTS,
    one_pol::RADIX_BITS};

  using scan_pol  = typename active_policy::ScanPolicy;
  const auto scan = ScanPolicy{
    ScanAlgorithm::lookback,
    ScanLookbackPolicy{
      scan_pol::BLOCK_THREADS,
      scan_pol::ITEMS_PER_THREAD,
      scan_pol::LOAD_ALGORITHM,
      scan_pol::LOAD_MODIFIER,
      scan_pol::STORE_ALGORITHM,
      scan_pol::SCAN_ALGORITHM,
      lookback_delay_policy_from_type<typename scan_pol::detail::delay_constructor_t>},
    {}};

  const auto downsweep     = radix_sort::convert_downsweep_policy(typename active_policy::DownsweepPolicy{});
  const auto alt_downsweep = radix_sort::convert_downsweep_policy(typename active_policy::AltDownsweepPolicy{});

  using up_pol = typename active_policy::UpsweepPolicy;
  const auto upsweep =
    RadixSortUpsweepPolicy{up_pol::BLOCK_THREADS, up_pol::ITEMS_PER_THREAD, up_pol::LOAD_MODIFIER, up_pol::RADIX_BITS};

  using alt_up_pol       = typename active_policy::AltUpsweepPolicy;
  const auto alt_upsweep = RadixSortUpsweepPolicy{
    alt_up_pol::BLOCK_THREADS, alt_up_pol::ITEMS_PER_THREAD, alt_up_pol::LOAD_MODIFIER, alt_up_pol::RADIX_BITS};

  const auto single_tile = radix_sort::convert_downsweep_policy(typename active_policy::SingleTilePolicy{});

  return RadixSortPolicy{
    active_policy::ONESWEEP ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass,
    histogram,
    exclusive_sum,
    onesweep,
    scan,
    downsweep,
    alt_downsweep,
    upsweep,
    alt_upsweep,
    single_tile};
}

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename LegacyActivePolicy>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE constexpr auto convert_policy(RadixSortPolicyWrapper<LegacyActivePolicy> policy)
  -> RadixSortPolicy
{
  return convert_policy<LegacyActivePolicy>();
}

// TODO(bgruber): remove in CCCL 4.0 when we drop the radix sort dispatcher after publishing the tuning API
template <typename PolicyHub>
struct policy_selector_from_hub
{
  _CCCL_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> RadixSortPolicy
  {
    return convert_policy<typename PolicyHub::MaxPolicy::ActivePolicy>();
  }
};

/**
 * @brief Tuning policy for kernel specialization
 *
 * @tparam KeyT
 *   Key type
 *
 * @tparam ValueT
 *   Value type
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 */
// TODO(bgruber): remove this in CCCL 4.0 when we remove the public radix sort dispatcher
template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_hub
{
  //------------------------------------------------------------------------------
  // Constants
  //------------------------------------------------------------------------------

  // Whether this is a keys-only (or key-value) sort
  static constexpr bool KEYS_ONLY = ::cuda::std::is_same_v<ValueT, NullType>;

  // Dominant-sized key/value type
  using DominantT = ::cuda::std::_If<(sizeof(ValueT) > sizeof(KeyT)), ValueT, KeyT>;

  //------------------------------------------------------------------------------
  // Architecture-specific tuning policies
  //------------------------------------------------------------------------------

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int PRIMARY_RADIX_BITS = (sizeof(KeyT) > 1) ? 7 : 5; // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5; // 3.1B 32b segmented keys/s (TitanX)
    static constexpr bool ONESWEEP              = false;
    static constexpr int ONESWEEP_RADIX_BITS    = 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      256,
      21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      160,
      39,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      31,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      11,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM60 (GP100)
  struct Policy600 : ChainedPolicy<600, Policy600, Policy500>
  {
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5; // 6.9B 32b keys/s (Quadro P100)
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5; // 5.9B 32b segmented keys/s (Quadro P100)
    static constexpr bool ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t); // 10.0B 32b keys/s (GP100, 64M random keys)
    static constexpr int ONESWEEP_RADIX_BITS = 8;
    static constexpr bool OFFSET_64BIT       = sizeof(OffsetT) == 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      256,
      OFFSET_64BIT ? 29 : 30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      25,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      OFFSET_64BIT ? 32 : 39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM61 (GP104)
  struct Policy610 : ChainedPolicy<610, Policy610, Policy600>
  {
    static constexpr int PRIMARY_RADIX_BITS = (sizeof(KeyT) > 1) ? 7 : 5; // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5; // 3.3B 32b segmented keys/s (1080)
    static constexpr bool ONESWEEP              = sizeof(KeyT) >= sizeof(uint32_t);
    static constexpr int ONESWEEP_RADIX_BITS    = 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      31,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      35,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = detail::agent_radix_sort_upsweep_policy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      detail::agent_radix_sort_upsweep_policy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM62 (Tegra, less RF)
  struct Policy620 : ChainedPolicy<620, Policy620, Policy610>
  {
    static constexpr int PRIMARY_RADIX_BITS  = 5;
    static constexpr int ALT_RADIX_BITS      = PRIMARY_RADIX_BITS - 1;
    static constexpr bool ONESWEEP           = sizeof(KeyT) >= sizeof(uint32_t);
    static constexpr int ONESWEEP_RADIX_BITS = 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      256,
      30,
      DominantT,
      2,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      ALT_RADIX_BITS>;

    // Upsweep policies
    using UpsweepPolicy    = DownsweepPolicy;
    using AltUpsweepPolicy = AltDownsweepPolicy;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy    = DownsweepPolicy;
    using AltSegmentedPolicy = AltDownsweepPolicy;
  };

  /// SM70 (GV100)
  struct Policy700 : ChainedPolicy<700, Policy700, Policy620>
  {
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5; // 7.62B 32b keys/s (GV100)
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5; // 8.7B 32b segmented keys/s (GV100)
    static constexpr bool ONESWEEP = sizeof(KeyT) >= sizeof(uint32_t); // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
    static constexpr int ONESWEEP_RADIX_BITS = 8;
    static constexpr bool OFFSET_64BIT       = sizeof(OffsetT) == 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      256,
      sizeof(KeyT) == 4 && sizeof(ValueT) == 4 ? 46 : 23,
      DominantT,
      4,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      OFFSET_64BIT ? 46 : 47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = detail::agent_radix_sort_upsweep_policy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = detail::
      agent_radix_sort_upsweep_policy<256, OFFSET_64BIT ? 46 : 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  /// SM80
  struct Policy800 : ChainedPolicy<800, Policy800, Policy700>
  {
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr bool ONESWEEP              = sizeof(KeyT) >= sizeof(uint32_t);
    static constexpr int ONESWEEP_RADIX_BITS    = 8;
    static constexpr bool OFFSET_64BIT          = sizeof(OffsetT) == 8;

    // Histogram policy
    using HistogramPolicy = detail::agent_radix_sort_histogram_policy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = detail::agent_radix_sort_onesweep_policy<
      384,
      OFFSET_64BIT && sizeof(KeyT) == 4 && !KEYS_ONLY ? 17 : 21,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    // ScanPolicy
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = detail::agent_radix_sort_upsweep_policy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      detail::agent_radix_sort_upsweep_policy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  template <typename OnesweepSmallKeyPolicySizes>
  struct OnesweepSmallKeyTunedPolicy
  {
    static constexpr bool ONESWEEP           = true;
    static constexpr int ONESWEEP_RADIX_BITS = 8;

    using HistogramPolicy    = detail::agent_radix_sort_histogram_policy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = detail::agent_radix_sort_exclusive_sum_policy<256, ONESWEEP_RADIX_BITS>;

  private:
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0;
    static constexpr int FLOAT_KEYS             = ::cuda::std::is_same_v<KeyT, float> ? 1 : 0;

    using OnesweepPolicyKey32 = detail::agent_radix_sort_onesweep_policy<
      384,
      KEYS_ONLY ? 20 - OFFSET_64BIT - FLOAT_KEYS
                : (sizeof(ValueT) < 8 ? (OFFSET_64BIT ? 17 : 23) : (OFFSET_64BIT ? 29 : 30)),
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepPolicyKey64 = detail::agent_radix_sort_onesweep_policy<
      384,
      sizeof(ValueT) < 8 ? 30 : 24,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepLargeKeyPolicy = ::cuda::std::_If<sizeof(KeyT) == 4, OnesweepPolicyKey32, OnesweepPolicyKey64>;

    using OnesweepSmallKeyPolicy = detail::agent_radix_sort_onesweep_policy<
      OnesweepSmallKeyPolicySizes::threads,
      OnesweepSmallKeyPolicySizes::items,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      8>;

  public:
    using OnesweepPolicy = ::cuda::std::_If<sizeof(KeyT) < 4, OnesweepSmallKeyPolicy, OnesweepLargeKeyPolicy>;

    // The Scan, Downsweep and Upsweep policies are never run on SM90, but we have to include them to prevent a
    // compilation error: When we compile e.g. for SM70 **and** SM90, the host compiler will reach calls to those
    // kernels, and instantiate them for MaxPolicy (which is Policy900) on the host, which will reach into the policies
    // below to set the launch bounds. The device compiler pass will also compile all kernels for SM70 **and** SM90,
    // even though only the Onesweep kernel is used on SM90.
    using ScanPolicy =
      agent_scan_policy<512,
                        23,
                        OffsetT,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    using AltDownsweepPolicy = detail::agent_radix_sort_downsweep_policy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy = detail::agent_radix_sort_upsweep_policy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      detail::agent_radix_sort_upsweep_policy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = detail::agent_radix_sort_downsweep_policy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;

    using AltSegmentedPolicy = detail::agent_radix_sort_downsweep_policy<
      384,
      11,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS - 1>;
  };

  struct Policy900
      : ChainedPolicy<900, Policy900, Policy800>
      , OnesweepSmallKeyTunedPolicy<sm90_small_key_tuning<sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>>
  {};

  struct Policy1000
      : ChainedPolicy<1000, Policy1000, Policy900>
      , OnesweepSmallKeyTunedPolicy<
          sm100_small_key_tuning<ValueT, sizeof(KeyT), KEYS_ONLY ? 0 : sizeof(ValueT), sizeof(OffsetT)>>
  {};

  using MaxPolicy = Policy1000;
};

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int __scale_num_parts(int nominal_4b_num_parts, int compute_t_size)
{
  return ::cuda::std::max(1, nominal_4b_num_parts * 4 / ::cuda::std::max(compute_t_size, 4));
}

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept radix_sort_policy_selector = detail::policy_selector<T, RadixSortPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;
  int value_size; // when 0, indicates keys-only
  int offset_size;
  type_t key_type;

  // Whether this is a keys-only (or key-value) sort
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int __keys_only() const
  {
    return value_size == 0;
  }

  // Dominant-sized key/value type
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int __dominant_size() const
  {
    return ::cuda::std::max(value_size, key_size);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
  make_onesweep_small_key_policy(const small_key_tuning_values& tuning) const -> RadixSortPolicy
  {
    const int primary_radix_bits     = (key_size > 1) ? 7 : 5;
    const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
    const int onesweep_radix_bits    = 8;

    const auto histogram = RadixSortHistogramPolicy{128, 16, __scale_num_parts(1, key_size), onesweep_radix_bits};

    const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

    const bool offset_64bit = offset_size == 8;
    const bool key_is_float = key_type == type_t::float32;

    const auto onesweep_policy_key32 = make_reg_scaled_radix_sort_onesweep_policy(
      384,
      __keys_only() ? 20 - offset_64bit - key_is_float
                    : (value_size < 8 ? (offset_64bit ? 17 : 23) : (offset_64bit ? 29 : 30)),
      __dominant_size(),
      RADIX_SORT_STORE_DIRECT,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      1,
      onesweep_radix_bits);

    const auto onesweep_policy_key64 = make_reg_scaled_radix_sort_onesweep_policy(
      384,
      value_size < 8 ? 30 : 24,
      __dominant_size(),
      RADIX_SORT_STORE_DIRECT,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      1,
      onesweep_radix_bits);

    const auto onesweep_large_key_policy = key_size == 4 ? onesweep_policy_key32 : onesweep_policy_key64;

    const auto onesweep_small_key_policy = make_reg_scaled_radix_sort_onesweep_policy(
      tuning.threads,
      tuning.items,
      __dominant_size(),
      RADIX_SORT_STORE_DIRECT,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      1,
      8);

    const auto onesweep = key_size < 4 ? onesweep_small_key_policy : onesweep_large_key_policy;

    // The scan, downsweep and upsweep policies are never run on SM90+, but we have to include them to prevent a
    // compilation error: When we compile e.g. for SM70 **and** SM90, the host compiler will reach calls to those
    // kernels, and instantiate them on the host, which will reach into the policies below to set the launch bounds. The
    // device compiler pass will also compile all kernels for SM70 **and** SM90, even though only the onesweep kernel is
    // used on SM90.

    const auto scan = make_mem_scaled_lookback_scan_policy(
      512,
      23,
      offset_size,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE,
      BLOCK_SCAN_RAKING_MEMOIZE);

    const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      512,
      23,
      __dominant_size(),
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      primary_radix_bits);

    const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      (key_size > 1) ? 256 : 128,
      47,
      __dominant_size(),
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      primary_radix_bits - 1);

    const auto upsweep =
      make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), LOAD_DEFAULT, primary_radix_bits);

    const auto alt_upsweep =
      make_reg_scaled_radix_sort_upsweep_policy(256, 47, __dominant_size(), LOAD_DEFAULT, primary_radix_bits - 1);

    const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      19,
      __dominant_size(),
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      single_tile_radix_bits);

    return RadixSortPolicy{
      RadixSortAlgorithm::onesweep,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      downsweep,
      alt_downsweep,
      upsweep,
      alt_upsweep,
      single_tile};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> RadixSortPolicy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      return make_onesweep_small_key_policy(get_sm100_tuning(key_size, value_size, offset_size, key_type));
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      return make_onesweep_small_key_policy(get_sm90_tuning(key_size, value_size, offset_size));
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5;
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      const auto use_onesweep =
        key_size >= int{sizeof(uint32_t)} ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
      const int onesweep_radix_bits = 8;
      const bool offset_64bit       = offset_size == 8;

      const auto histogram = RadixSortHistogramPolicy{128, 16, __scale_num_parts(1, key_size), onesweep_radix_bits};

      const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        384,
        offset_64bit && key_size == 4 && !__keys_only() ? 17 : 21,
        __dominant_size(),
        RADIX_SORT_STORE_DIRECT,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_RAKING_MEMOIZE,
        1,
        onesweep_radix_bits);

      const auto scan = make_mem_scaled_lookback_scan_policy(
        512,
        23,
        offset_size,
        BLOCK_LOAD_WARP_TRANSPOSE,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        512,
        23,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        (key_size > 1) ? 256 : 128,
        47,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits - 1);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), LOAD_DEFAULT, primary_radix_bits);

      const auto alt_upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 47, __dominant_size(), LOAD_DEFAULT, primary_radix_bits - 1);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        single_tile_radix_bits);

      return RadixSortPolicy{
        use_onesweep,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile};
    }

    if (cc >= ::cuda::compute_capability{7, 0})
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 7.62B 32b keys/s (GV100)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
      const auto use_onesweep =
        key_size >= int{sizeof(uint32_t)} ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
      const int onesweep_radix_bits = 8;
      const bool offset_64bit       = offset_size == 8;

      const auto histogram = RadixSortHistogramPolicy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        key_size == 4 && value_size == 4 ? 46 : 23,
        __dominant_size(),
        RADIX_SORT_STORE_DIRECT,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        4,
        onesweep_radix_bits);

      const auto scan = make_mem_scaled_lookback_scan_policy(
        512,
        23,
        offset_size,
        BLOCK_LOAD_WARP_TRANSPOSE,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        512,
        23,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        (key_size > 1) ? 256 : 128,
        offset_64bit ? 46 : 47,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits - 1);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), LOAD_DEFAULT, primary_radix_bits);

      const auto alt_upsweep = make_reg_scaled_radix_sort_upsweep_policy(
        256, offset_64bit ? 46 : 47, __dominant_size(), LOAD_DEFAULT, primary_radix_bits - 1);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        single_tile_radix_bits);

      return RadixSortPolicy{
        use_onesweep,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile};
    }

    if (cc >= ::cuda::compute_capability{6, 2})
    {
      const int primary_radix_bits = 5;
      const int alt_radix_bits     = primary_radix_bits - 1;
      const auto use_onesweep =
        key_size >= int{sizeof(uint32_t)} ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
      const int onesweep_radix_bits = 8;

      const auto histogram = RadixSortHistogramPolicy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        30,
        __dominant_size(),
        RADIX_SORT_STORE_DIRECT,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        2,
        onesweep_radix_bits);

      const auto scan = make_mem_scaled_lookback_scan_policy(
        512,
        23,
        offset_size,
        BLOCK_LOAD_WARP_TRANSPOSE,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        16,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE,
        primary_radix_bits);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        16,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE,
        alt_radix_bits);

      const auto upsweep = RadixSortUpsweepPolicy{
        downsweep.threads_per_block, downsweep.items_per_thread, downsweep.load_modifier, downsweep.radix_bits};

      const auto alt_upsweep = RadixSortUpsweepPolicy{
        alt_downsweep.threads_per_block,
        alt_downsweep.items_per_thread,
        alt_downsweep.load_modifier,
        alt_downsweep.radix_bits};

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits);

      return RadixSortPolicy{
        use_onesweep,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile};
    }

    if (cc >= ::cuda::compute_capability{6, 1})
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      // 10.0B 32b keys/s (GP100, 64M random keys)
      const auto use_onesweep =
        key_size >= int{sizeof(uint32_t)} ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
      const int onesweep_radix_bits = 8;

      const auto histogram = RadixSortHistogramPolicy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        30,
        __dominant_size(),
        RADIX_SORT_STORE_DIRECT,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        2,
        onesweep_radix_bits);

      const auto scan = make_mem_scaled_lookback_scan_policy(
        512,
        23,
        offset_size,
        BLOCK_LOAD_WARP_TRANSPOSE,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        31,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_RAKING_MEMOIZE,
        primary_radix_bits);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        35,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE,
        primary_radix_bits - 1);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(128, 16, __dominant_size(), LOAD_LDG, primary_radix_bits);

      const auto alt_upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(128, 16, __dominant_size(), LOAD_LDG, primary_radix_bits - 1);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        single_tile_radix_bits);

      return RadixSortPolicy{
        use_onesweep,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 6.9B 32b keys/s (Quadro P100)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      // 10.0B 32b keys/s (GP100, 64M random keys)
      const auto use_onesweep =
        key_size >= int{sizeof(uint32_t)} ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
      const int onesweep_radix_bits = 8;
      const bool offset_64bit       = (offset_size == 8);

      const auto histogram = RadixSortHistogramPolicy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        offset_64bit ? 29 : 30,
        __dominant_size(),
        RADIX_SORT_STORE_DIRECT,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        2,
        onesweep_radix_bits);

      const auto scan = make_mem_scaled_lookback_scan_policy(
        512,
        23,
        offset_size,
        BLOCK_LOAD_WARP_TRANSPOSE,
        LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        25,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        offset_64bit ? 32 : 39,
        __dominant_size(),
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        primary_radix_bits - 1);

      const auto upsweep = RadixSortUpsweepPolicy{
        downsweep.threads_per_block, downsweep.items_per_thread, downsweep.load_modifier, downsweep.radix_bits};

      const auto alt_upsweep = RadixSortUpsweepPolicy{
        alt_downsweep.threads_per_block,
        alt_downsweep.items_per_thread,
        alt_downsweep.load_modifier,
        alt_downsweep.radix_bits};

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS,
        single_tile_radix_bits);

      return RadixSortPolicy{
        use_onesweep,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile};
    }

    // SM50
    const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
    const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
    const auto use_onesweep          = RadixSortAlgorithm::multi_pass;
    const int onesweep_radix_bits    = 8;

    const auto histogram = RadixSortHistogramPolicy{256, 8, __scale_num_parts(1, key_size), onesweep_radix_bits};

    const auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

    const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
      256,
      21,
      __dominant_size(),
      RADIX_SORT_STORE_DIRECT,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      1,
      onesweep_radix_bits);

    const auto scan = make_mem_scaled_lookback_scan_policy(
      512,
      23,
      offset_size,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE,
      BLOCK_SCAN_RAKING_MEMOIZE);

    const auto downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      160,
      39,
      __dominant_size(),
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      primary_radix_bits);

    const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      16,
      __dominant_size(),
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      primary_radix_bits - 1);

    const auto upsweep = RadixSortUpsweepPolicy{
      downsweep.threads_per_block, downsweep.items_per_thread, downsweep.load_modifier, downsweep.radix_bits};

    const auto alt_upsweep = RadixSortUpsweepPolicy{
      alt_downsweep.threads_per_block,
      alt_downsweep.items_per_thread,
      alt_downsweep.load_modifier,
      alt_downsweep.radix_bits};

    const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      19,
      __dominant_size(),
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      single_tile_radix_bits);

    return RadixSortPolicy{
      use_onesweep,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      downsweep,
      alt_downsweep,
      upsweep,
      alt_upsweep,
      single_tile};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(radix_sort_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const -> RadixSortPolicy
  {
    constexpr auto policies = policy_selector{
      int{sizeof(KeyT)},
      ::cuda::std::is_same_v<ValueT, NullType> ? 0 : int{sizeof(ValueT)},
      int{sizeof(OffsetT)},
      classify_type<KeyT>};
    return policies(cc);
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
