// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/optional>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN
namespace detail::radix_sort
{
enum class delay_constructor_kind
{
  no_delay,
  fixed_delay,
  exponential_backoff,
  exponential_backoff_jitter,
  exponential_backoff_jitter_window,
  exponential_backon_jitter_window,
  exponential_backon_jitter,
  exponential_backon
};

#if !_CCCL_COMPILER(NVRTC)
inline ::std::ostream& operator<<(::std::ostream& os, delay_constructor_kind kind)
{
  switch (kind)
  {
    case delay_constructor_kind::no_delay:
      return os << "delay_constructor_kind::no_delay";
    case delay_constructor_kind::fixed_delay:
      return os << "delay_constructor_kind::fixed_delay";
    case delay_constructor_kind::exponential_backoff:
      return os << "delay_constructor_kind::exponential_backoff";
    case delay_constructor_kind::exponential_backoff_jitter:
      return os << "delay_constructor_kind::exponential_backoff_jitter";
    case delay_constructor_kind::exponential_backoff_jitter_window:
      return os << "delay_constructor_kind::exponential_backoff_jitter_window";
    case delay_constructor_kind::exponential_backon_jitter_window:
      return os << "delay_constructor_kind::exponential_backon_jitter_window";
    case delay_constructor_kind::exponential_backon_jitter:
      return os << "delay_constructor_kind::exponential_backon_jitter";
    case delay_constructor_kind::exponential_backon:
      return os << "delay_constructor_kind::exponential_backon";
    default:
      return os << "<unknown delay_constructor_kind: " << static_cast<int>(kind) << ">";
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

struct delay_constructor_policy
{
  delay_constructor_kind kind;
  unsigned int delay;
  unsigned int l2_write_latency;

  _CCCL_API constexpr friend bool operator==(const delay_constructor_policy& lhs, const delay_constructor_policy& rhs)
  {
    return lhs.kind == rhs.kind && lhs.delay == rhs.delay && lhs.l2_write_latency == rhs.l2_write_latency;
  }

  _CCCL_API constexpr friend bool operator!=(const delay_constructor_policy& lhs, const delay_constructor_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const delay_constructor_policy& p)
  {
    return os << "delay_constructor_policy { .kind = " << p.kind << ", .delay = " << p.delay
              << ", .l2_write_latency = " << p.l2_write_latency << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

template <typename DelayConstructor>
inline constexpr auto delay_constructor_policy_from_type = 0;

template <unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<no_delay_constructor_t<L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::no_delay, 0, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<fixed_delay_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::fixed_delay, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backoff_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backoff, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backoff_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backoff_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backon_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backon, Delay, L2WriteLatency};

// TODO(bgruber): this is modeled after <look_back_helper.cuh>, unify this
template <delay_constructor_kind Kind, unsigned int Delay, unsigned int L2WriteLatency>
struct __delay_constructor_t_helper
{
private:
  using delay_constructors = ::cuda::std::__type_list<
    detail::no_delay_constructor_t<L2WriteLatency>,
    detail::fixed_delay_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backoff_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>,
    detail::exponential_backon_constructor_t<Delay, L2WriteLatency>>;

public:
  using type = ::cuda::std::__type_at_c<static_cast<int>(Kind), delay_constructors>;
};

template <delay_constructor_kind Kind, unsigned int Delay, unsigned int L2WriteLatency>
using delay_constructor_t = typename __delay_constructor_t_helper<Kind, Delay, L2WriteLatency>::type;

struct radix_sort_histogram_policy
{
  int block_threads;
  int items_per_thread;
  int num_parts;
  int radix_bits;

  _CCCL_API constexpr friend bool
  operator==(const radix_sort_histogram_policy& lhs, const radix_sort_histogram_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.num_parts == rhs.num_parts && lhs.radix_bits == rhs.radix_bits;
  }

  _CCCL_API constexpr friend bool
  operator!=(const radix_sort_histogram_policy& lhs, const radix_sort_histogram_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_histogram_policy& p)
  {
    return os << "radix_sort_histogram_policy { .block_threads = " << p.block_threads << ", .items_per_thread = "
              << p.items_per_thread << ", .num_parts = " << p.num_parts << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct radix_sort_exclusive_sum_policy
{
  int block_threads;
  int radix_bits;

  _CCCL_API constexpr friend bool
  operator==(const radix_sort_exclusive_sum_policy& lhs, const radix_sort_exclusive_sum_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.radix_bits == rhs.radix_bits;
  }

  _CCCL_API constexpr friend bool
  operator!=(const radix_sort_exclusive_sum_policy& lhs, const radix_sort_exclusive_sum_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_exclusive_sum_policy& p)
  {
    return os << "radix_sort_exclusive_sum_policy { .block_threads = " << p.block_threads
              << ", .radix_bits = " << p.radix_bits << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct radix_sort_onesweep_policy
{
  int block_threads;
  int items_per_thread;
  int rank_num_parts;
  int radix_bits;
  RadixRankAlgorithm rank_algorith;
  BlockScanAlgorithm scan_algorithm;
  RadixSortStoreAlgorithm store_algorithm;

  _CCCL_API constexpr friend bool
  operator==(const radix_sort_onesweep_policy& lhs, const radix_sort_onesweep_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.rank_num_parts == rhs.rank_num_parts && lhs.radix_bits == rhs.radix_bits
        && lhs.rank_algorith == rhs.rank_algorith && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.store_algorithm == rhs.store_algorithm;
  }

  _CCCL_API constexpr friend bool
  operator!=(const radix_sort_onesweep_policy& lhs, const radix_sort_onesweep_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_onesweep_policy& p)
  {
    return os
        << "radix_sort_onesweep_policy { .block_threads = " << p.block_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .rank_num_parts = " << p.rank_num_parts
        << ", .radix_bits = " << p.radix_bits << ", .rank_algorith = " << p.rank_algorith
        << ", .scan_algorithm = " << p.scan_algorithm << ", .store_algorithm = " << p.store_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_API constexpr auto make_reg_scaled_radix_sort_onesweep_policy(
  int nominal_4b_block_threads,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  int rank_num_parts,
  int radix_bits,
  RadixRankAlgorithm rank_algorith,
  BlockScanAlgorithm scan_algorithm,
  RadixSortStoreAlgorithm store_algorithm) -> radix_sort_onesweep_policy
{
  const auto scaled = scale_reg_bound(nominal_4b_block_threads, nominal_4b_items_per_thread, compute_t_size);
  return radix_sort_onesweep_policy{
    scaled.block_threads,
    scaled.items_per_thread,
    rank_num_parts,
    radix_bits,
    rank_algorith,
    scan_algorithm,
    store_algorithm};
}

// TODO(bgruber): move this into the scan tuning header
struct scan_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor;

  _CCCL_API constexpr friend bool operator==(const scan_policy& lhs, const scan_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.delay_constructor == rhs.delay_constructor;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_policy& lhs, const scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_policy& p)
  {
    return os
        << "scan_policy { .block_threads = " << p.block_threads << ", .items_per_thread = " << p.items_per_thread
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .store_algorithm = " << p.store_algorithm << ", .scan_algorithm = " << p.scan_algorithm
        << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_API constexpr auto make_mem_scaled_scan_policy(
  int nominal_4b_block_threads,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  BlockLoadAlgorithm load_algorithm,
  CacheLoadModifier load_modifier,
  BlockStoreAlgorithm store_algorithm,
  BlockScanAlgorithm scan_algorithm,
  delay_constructor_policy delay_constructor = {delay_constructor_kind::fixed_delay, 350, 450}) -> scan_policy
{
  const auto scaled = scale_mem_bound(nominal_4b_block_threads, nominal_4b_items_per_thread, compute_t_size);
  return scan_policy{
    scaled.block_threads,
    scaled.items_per_thread,
    load_algorithm,
    load_modifier,
    store_algorithm,
    scan_algorithm,
    delay_constructor};
}

struct radix_sort_downsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  RadixRankAlgorithm rank_algorithm;
  BlockScanAlgorithm scan_algorithm;

  _CCCL_API constexpr friend bool
  operator==(const radix_sort_downsweep_policy& lhs, const radix_sort_downsweep_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.radix_bits == rhs.radix_bits && lhs.load_algorithm == rhs.load_algorithm
        && lhs.load_modifier == rhs.load_modifier && lhs.rank_algorithm == rhs.rank_algorithm
        && lhs.scan_algorithm == rhs.scan_algorithm;
  }

  _CCCL_API constexpr friend bool
  operator!=(const radix_sort_downsweep_policy& lhs, const radix_sort_downsweep_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_downsweep_policy& p)
  {
    return os
        << "radix_sort_downsweep_policy { .block_threads = " << p.block_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .radix_bits = " << p.radix_bits
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .rank_algorithm = " << p.rank_algorithm << ", .scan_algorithm = " << p.scan_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_API constexpr auto make_reg_scaled_radix_sort_downsweep_policy(
  int nominal_4b_block_threads,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  int radix_bits,
  BlockLoadAlgorithm load_algorithm,
  CacheLoadModifier load_modifier,
  RadixRankAlgorithm rank_algorithm,
  BlockScanAlgorithm scan_algorithm) -> radix_sort_downsweep_policy
{
  const auto scaled = scale_reg_bound(nominal_4b_block_threads, nominal_4b_items_per_thread, compute_t_size);
  return radix_sort_downsweep_policy{
    scaled.block_threads,
    scaled.items_per_thread,
    radix_bits,
    load_algorithm,
    load_modifier,
    rank_algorithm,
    scan_algorithm};
}

struct radix_sort_upsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;
  CacheLoadModifier load_modifier;

  _CCCL_API constexpr friend bool operator==(const radix_sort_upsweep_policy& lhs, const radix_sort_upsweep_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.radix_bits == rhs.radix_bits && lhs.load_modifier == rhs.load_modifier;
  }

  _CCCL_API constexpr friend bool operator!=(const radix_sort_upsweep_policy& lhs, const radix_sort_upsweep_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_upsweep_policy& p)
  {
    return os
        << "radix_sort_upsweep_policy { .block_threads = " << p.block_threads << ", .items_per_thread = "
        << p.items_per_thread << ", .radix_bits = " << p.radix_bits << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_API constexpr auto make_reg_scaled_radix_sort_upsweep_policy(
  int nominal_4b_block_threads,
  int nominal_4b_items_per_thread,
  int compute_t_size,
  int radix_bits,
  CacheLoadModifier load_modifier) -> radix_sort_upsweep_policy
{
  const auto scaled = scale_reg_bound(nominal_4b_block_threads, nominal_4b_items_per_thread, compute_t_size);
  return radix_sort_upsweep_policy{scaled.block_threads, scaled.items_per_thread, radix_bits, load_modifier};
}

struct radix_sort_policy
{
  bool use_onesweep;
  int onesweep_radix_bits;
  radix_sort_histogram_policy histogram;
  radix_sort_exclusive_sum_policy exclusive_sum;
  radix_sort_onesweep_policy onesweep;
  scan_policy scan;
  radix_sort_downsweep_policy downsweep;
  radix_sort_downsweep_policy alt_downsweep;
  radix_sort_upsweep_policy upsweep;
  radix_sort_upsweep_policy alt_upsweep;
  radix_sort_downsweep_policy single_tile;
  // TODO(bgruber): move those over to segmented radix sort when we port it
  radix_sort_downsweep_policy segmented;
  radix_sort_downsweep_policy alt_segmented;

  _CCCL_API constexpr friend bool operator==(const radix_sort_policy& lhs, const radix_sort_policy& rhs)
  {
    return lhs.use_onesweep == rhs.use_onesweep && lhs.onesweep_radix_bits == rhs.onesweep_radix_bits
        && lhs.histogram == rhs.histogram && lhs.exclusive_sum == rhs.exclusive_sum && lhs.onesweep == rhs.onesweep
        && lhs.scan == rhs.scan && lhs.downsweep == rhs.downsweep && lhs.alt_downsweep == rhs.alt_downsweep
        && lhs.upsweep == rhs.upsweep && lhs.alt_upsweep == rhs.alt_upsweep && lhs.single_tile == rhs.single_tile
        && lhs.segmented == rhs.segmented && lhs.alt_segmented == rhs.alt_segmented;
  }

  _CCCL_API constexpr friend bool operator!=(const radix_sort_policy& lhs, const radix_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const radix_sort_policy& p)
  {
    return os
        << "radix_sort_policy { .use_onesweep = " << p.use_onesweep
        << ", .onesweep_radix_bits = " << p.onesweep_radix_bits << ", .histogram = " << p.histogram
        << ", .exclusive_sum = " << p.exclusive_sum << ", .onesweep = " << p.onesweep << ", .scan = " << p.scan
        << ", .downsweep = " << p.downsweep << ", .alt_downsweep = " << p.alt_downsweep << ", .upsweep = " << p.upsweep
        << ", .alt_upsweep = " << p.alt_upsweep << ", .single_tile = " << p.single_tile
        << ", .segmented = " << p.segmented << ", .alt_segmented = " << p.alt_segmented << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

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

_CCCL_API constexpr auto get_sm90_tuning(int key_size, int value_size, int offset_size) -> small_key_tuning_values
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

_CCCL_API constexpr auto get_sm100_tuning(int key_size, int value_size, int offset_size, type_t key_type)
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

// TODO(bgruber): remove when segmented radix sort is ported to the new tuning API
template <typename PolicyT, typename = void>
struct RadixSortPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE RadixSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

#if defined(CUB_DEFINE_RUNTIME_POLICIES) || defined(CUB_ENABLE_POLICY_PTX_JSON)
using namespace radix_sort_runtime_policies;
#endif

// TODO(bgruber): remove when segmented radix sort is ported to the new tuning API
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
  _CCCL_HOST_DEVICE static constexpr int BlockThreads(PolicyT /*policy*/)
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

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<
      key<"SingleTilePolicy">()     = SingleTile().EncodedPolicy(),
      key<"OnesweepPolicy">()       = Onesweep().EncodedPolicy(),
      key<"UpsweepPolicy">()        = Upsweep().EncodedPolicy(),
      key<"AltUpsweepPolicy">()     = AltUpsweep().EncodedPolicy(),
      key<"DownsweepPolicy">()      = Downsweep().EncodedPolicy(),
      key<"AltDownsweepPolicy">()   = AltDownsweep().EncodedPolicy(),
      key<"HistogramPolicy">()      = Histogram().EncodedPolicy(),
      key<"ScanPolicy">()           = Scan().EncodedPolicy(),
      key<"ScanDelayConstructor">() = StaticPolicyT::ScanPolicy::detail::delay_constructor_t::EncodedConstructor(),
      key<"ExclusiveSumPolicy">()   = ExclusiveSum().EncodedPolicy(),
      key<"Onesweep">()             = value<IsOnesweep()>()>();
  }
#endif
};

// TODO(bgruber): remove when segmented radix sort is ported to the new tuning API
template <typename PolicyT>
_CCCL_HOST_DEVICE RadixSortPolicyWrapper<PolicyT> MakeRadixSortPolicyWrapper(PolicyT policy)
{
  return RadixSortPolicyWrapper<PolicyT>{policy};
}

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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      160,
      39,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
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
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      31,
      DominantT,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      25,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
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
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      384,
      31,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      35,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<128, 16, DominantT, LOAD_LDG, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      256,
      16,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
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
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<256, 8, 8, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      OFFSET_64BIT ? 46 : 47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy =
      AgentRadixSortUpsweepPolicy<256, OFFSET_64BIT ? 46 : 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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
    using HistogramPolicy = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;

    // Exclusive sum policy
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

    // Onesweep policy
    using OnesweepPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    // Downsweep policies
    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;
    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    // Upsweep policies
    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    // Single-tile policy
    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    // Segmented policies
    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;
    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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

    using HistogramPolicy    = AgentRadixSortHistogramPolicy<128, 16, 1, KeyT, ONESWEEP_RADIX_BITS>;
    using ExclusiveSumPolicy = AgentRadixSortExclusiveSumPolicy<256, ONESWEEP_RADIX_BITS>;

  private:
    static constexpr int PRIMARY_RADIX_BITS     = (sizeof(KeyT) > 1) ? 7 : 5;
    static constexpr int SINGLE_TILE_RADIX_BITS = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int SEGMENTED_RADIX_BITS   = (sizeof(KeyT) > 1) ? 6 : 5;
    static constexpr int OFFSET_64BIT           = sizeof(OffsetT) == 8 ? 1 : 0;
    static constexpr int FLOAT_KEYS             = ::cuda::std::is_same_v<KeyT, float> ? 1 : 0;

    using OnesweepPolicyKey32 = AgentRadixSortOnesweepPolicy<
      384,
      KEYS_ONLY ? 20 - OFFSET_64BIT - FLOAT_KEYS
                : (sizeof(ValueT) < 8 ? (OFFSET_64BIT ? 17 : 23) : (OFFSET_64BIT ? 29 : 30)),
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepPolicyKey64 = AgentRadixSortOnesweepPolicy<
      384,
      sizeof(ValueT) < 8 ? 30 : 24,
      DominantT,
      1,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT,
      ONESWEEP_RADIX_BITS>;

    using OnesweepLargeKeyPolicy = ::cuda::std::_If<sizeof(KeyT) == 4, OnesweepPolicyKey32, OnesweepPolicyKey64>;

    using OnesweepSmallKeyPolicy = AgentRadixSortOnesweepPolicy<
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
      AgentScanPolicy<512,
                      23,
                      OffsetT,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_RAKING_MEMOIZE>;

    using DownsweepPolicy = AgentRadixSortDownsweepPolicy<
      512,
      23,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS>;

    using AltDownsweepPolicy = AgentRadixSortDownsweepPolicy<
      (sizeof(KeyT) > 1) ? 256 : 128,
      47,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      PRIMARY_RADIX_BITS - 1>;

    using UpsweepPolicy    = AgentRadixSortUpsweepPolicy<256, 23, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS>;
    using AltUpsweepPolicy = AgentRadixSortUpsweepPolicy<256, 47, DominantT, LOAD_DEFAULT, PRIMARY_RADIX_BITS - 1>;

    using SingleTilePolicy = AgentRadixSortDownsweepPolicy<
      256,
      19,
      DominantT,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SINGLE_TILE_RADIX_BITS>;

    using SegmentedPolicy = AgentRadixSortDownsweepPolicy<
      192,
      39,
      DominantT,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS,
      SEGMENTED_RADIX_BITS>;

    using AltSegmentedPolicy = AgentRadixSortDownsweepPolicy<
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

[[nodiscard]] _CCCL_API constexpr int __scale_num_parts(int nominal_4b_num_parts, int compute_t_size)
{
  return ::cuda::std::max(1, nominal_4b_num_parts * 4 / ::cuda::std::max(compute_t_size, 4));
}

struct policy_selector
{
  int key_size;
  int value_size; // when 0, indicates keys-only
  int offset_size;
  type_t key_type;

  // Whether this is a keys-only (or key-value) sort
  [[nodiscard]] _CCCL_API constexpr int __keys_only() const
  {
    return value_size == 0;
  }

  // Dominant-sized key/value type
  [[nodiscard]] _CCCL_API constexpr int __dominant_size() const
  {
    return ::cuda::std::max(value_size, key_size);
  }

  [[nodiscard]] _CCCL_API constexpr auto make_onsweep_small_key_policy(const small_key_tuning_values& tuning) const
    -> radix_sort_policy
  {
    const int primary_radix_bits     = (key_size > 1) ? 7 : 5;
    const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
    const int segmented_radix_bits   = (key_size > 1) ? 6 : 5;
    const int onesweep_radix_bits    = 8;

    const auto histogram = radix_sort_histogram_policy{128, 16, __scale_num_parts(1, key_size), onesweep_radix_bits};

    const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

    const bool offset_64bit = offset_size == 8;
    const bool key_is_float = key_type == type_t::float32;

    const auto onesweep_policy_key32 = make_reg_scaled_radix_sort_onesweep_policy(
      384,
      __keys_only() ? 20 - offset_64bit - key_is_float
                    : (value_size < 8 ? (offset_64bit ? 17 : 23) : (offset_64bit ? 29 : 30)),
      __dominant_size(),
      1,
      onesweep_radix_bits,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT);

    const auto onesweep_policy_key64 = make_reg_scaled_radix_sort_onesweep_policy(
      384,
      value_size < 8 ? 30 : 24,
      __dominant_size(),
      1,
      onesweep_radix_bits,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT);

    const auto onesweep_large_key_policy = key_size == 4 ? onesweep_policy_key32 : onesweep_policy_key64;

    const auto onesweep_small_key_policy = make_reg_scaled_radix_sort_onesweep_policy(
      tuning.threads,
      tuning.items,
      __dominant_size(),
      1,
      8,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_RAKING_MEMOIZE,
      RADIX_SORT_STORE_DIRECT);

    const auto onesweep = key_size < 4 ? onesweep_small_key_policy : onesweep_large_key_policy;

    // The scan, downsweep and upsweep policies are never run on SM90+, but we have to include them to prevent a
    // compilation error: When we compile e.g. for SM70 **and** SM90, the host compiler will reach calls to those
    // kernels, and instantiate them on the host, which will reach into the policies below to set the launch bounds. The
    // device compiler pass will also compile all kernels for SM70 **and** SM90, even though only the onesweep kernel is
    // used on SM90.

    const auto scan = make_mem_scaled_scan_policy(
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
      primary_radix_bits,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MATCH,
      BLOCK_SCAN_WARP_SCANS);

    const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      (key_size > 1) ? 256 : 128,
      47,
      __dominant_size(),
      primary_radix_bits - 1,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto upsweep =
      make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), primary_radix_bits, LOAD_DEFAULT);

    const auto alt_upsweep =
      make_reg_scaled_radix_sort_upsweep_policy(256, 47, __dominant_size(), primary_radix_bits - 1, LOAD_DEFAULT);

    const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      19,
      __dominant_size(),
      single_tile_radix_bits,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
      192,
      39,
      __dominant_size(),
      segmented_radix_bits,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
      384,
      11,
      __dominant_size(),
      segmented_radix_bits - 1,
      BLOCK_LOAD_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    return radix_sort_policy{
      /* use_onesweep */ true,
      onesweep_radix_bits,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      downsweep,
      alt_downsweep,
      upsweep,
      alt_upsweep,
      single_tile,
      segmented,
      alt_segmented};
  }

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> radix_sort_policy
  {
    // TODO(bgruber): we should probably separate the segmented policies and move them somewhere else

    if (arch >= ::cuda::arch_id::sm_100)
    {
      return make_onsweep_small_key_policy(get_sm100_tuning(key_size, value_size, offset_size, key_type));
    }

    if (arch >= ::cuda::arch_id::sm_90)
    {
      return make_onsweep_small_key_policy(get_sm90_tuning(key_size, value_size, offset_size));
    }

    if (arch >= ::cuda::arch_id::sm_80)
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5;
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      const int segmented_radix_bits   = (key_size > 1) ? 6 : 5;
      const bool use_onesweep          = key_size >= int{sizeof(uint32_t)};
      const int onesweep_radix_bits    = 8;
      const bool offset_64bit          = offset_size == 8;

      const auto histogram = radix_sort_histogram_policy{128, 16, __scale_num_parts(1, key_size), onesweep_radix_bits};

      const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        384,
        offset_64bit && key_size == 4 && !__keys_only() ? 17 : 21,
        __dominant_size(),
        1,
        onesweep_radix_bits,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_RAKING_MEMOIZE,
        RADIX_SORT_STORE_DIRECT);

      const auto scan = make_mem_scaled_scan_policy(
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
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        (key_size > 1) ? 256 : 128,
        47,
        __dominant_size(),
        primary_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), primary_radix_bits, LOAD_DEFAULT);

      const auto alt_upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 47, __dominant_size(), primary_radix_bits - 1, LOAD_DEFAULT);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        single_tile_radix_bits,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return radix_sort_policy{
        use_onesweep,
        onesweep_radix_bits,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile,
        segmented,
        alt_segmented};
    }

    if (arch >= ::cuda::arch_id::sm_70)
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 7.62B 32b keys/s (GV100)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      const int segmented_radix_bits   = (key_size > 1) ? 6 : 5; // 8.7B 32b segmented keys/s (GV100)
      const bool use_onesweep = key_size >= int{sizeof(uint32_t)}; // 15.8B 32b keys/s (V100-SXM2, 64M random keys)
      const int onesweep_radix_bits = 8;
      const bool offset_64bit       = offset_size == 8;

      const auto histogram = radix_sort_histogram_policy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        key_size == 4 && value_size == 4 ? 46 : 23,
        __dominant_size(),
        4,
        onesweep_radix_bits,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        RADIX_SORT_STORE_DIRECT);

      const auto scan = make_mem_scaled_scan_policy(
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
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        (key_size > 1) ? 256 : 128,
        offset_64bit ? 46 : 47,
        __dominant_size(),
        primary_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(256, 23, __dominant_size(), primary_radix_bits, LOAD_DEFAULT);

      const auto alt_upsweep = make_reg_scaled_radix_sort_upsweep_policy(
        256, offset_64bit ? 46 : 47, __dominant_size(), primary_radix_bits - 1, LOAD_DEFAULT);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        single_tile_radix_bits,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return radix_sort_policy{
        use_onesweep,
        onesweep_radix_bits,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile,
        segmented,
        alt_segmented};
    }

    if (static_cast<int>(arch) >= 62) // TODO(bgruber): add ::cuda::arch_id::sm_62
    {
      const int primary_radix_bits  = 5;
      const int alt_radix_bits      = primary_radix_bits - 1;
      const bool use_onesweep       = key_size >= int{sizeof(uint32_t)};
      const int onesweep_radix_bits = 8;

      const auto histogram = radix_sort_histogram_policy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        30,
        __dominant_size(),
        2,
        onesweep_radix_bits,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        RADIX_SORT_STORE_DIRECT);

      const auto scan = make_mem_scaled_scan_policy(
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
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        16,
        __dominant_size(),
        alt_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto upsweep = radix_sort_upsweep_policy{
        downsweep.block_threads, downsweep.items_per_thread, downsweep.radix_bits, downsweep.load_modifier};

      const auto alt_upsweep = radix_sort_upsweep_policy{
        alt_downsweep.block_threads,
        alt_downsweep.items_per_thread,
        alt_downsweep.radix_bits,
        alt_downsweep.load_modifier};

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        primary_radix_bits,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto segmented     = downsweep;
      const auto alt_segmented = alt_downsweep;

      return radix_sort_policy{
        use_onesweep,
        onesweep_radix_bits,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile,
        segmented,
        alt_segmented};
    }

    if (arch >= ::cuda::arch_id::sm_61)
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 3.4B 32b keys/s, 1.83B 32b pairs/s (1080)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      const int segmented_radix_bits   = (key_size > 1) ? 6 : 5; // 3.3B 32b segmented keys/s (1080)
      const bool use_onesweep          = key_size >= int{sizeof(uint32_t)}; // 10.0B 32b keys/s (GP100, 64M random keys)
      const int onesweep_radix_bits    = 8;

      const auto histogram = radix_sort_histogram_policy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        30,
        __dominant_size(),
        2,
        onesweep_radix_bits,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        RADIX_SORT_STORE_DIRECT);

      const auto scan = make_mem_scaled_scan_policy(
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
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        35,
        __dominant_size(),
        primary_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(128, 16, __dominant_size(), primary_radix_bits, LOAD_LDG);

      const auto alt_upsweep =
        make_reg_scaled_radix_sort_upsweep_policy(128, 16, __dominant_size(), primary_radix_bits - 1, LOAD_LDG);

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        single_tile_radix_bits,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return radix_sort_policy{
        use_onesweep,
        onesweep_radix_bits,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile,
        segmented,
        alt_segmented};
    }

    if (arch >= ::cuda::arch_id::sm_60)
    {
      const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 6.9B 32b keys/s (Quadro P100)
      const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
      const int segmented_radix_bits   = (key_size > 1) ? 6 : 5; // 5.9B 32b segmented keys/s (Quadro P100)
      const bool use_onesweep          = key_size >= int{sizeof(uint32_t)}; // 10.0B 32b keys/s (GP100, 64M random keys)
      const int onesweep_radix_bits    = 8;
      const bool offset_64bit          = (offset_size == 8);

      const auto histogram = radix_sort_histogram_policy{256, 8, __scale_num_parts(8, key_size), onesweep_radix_bits};

      const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

      const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
        256,
        offset_64bit ? 29 : 30,
        __dominant_size(),
        2,
        onesweep_radix_bits,
        RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
        BLOCK_SCAN_WARP_SCANS,
        RADIX_SORT_STORE_DIRECT);

      const auto scan = make_mem_scaled_scan_policy(
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
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MATCH,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        offset_64bit ? 32 : 39,
        __dominant_size(),
        primary_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto upsweep = radix_sort_upsweep_policy{
        downsweep.block_threads, downsweep.items_per_thread, downsweep.radix_bits, downsweep.load_modifier};

      const auto alt_upsweep = radix_sort_upsweep_policy{
        alt_downsweep.block_threads,
        alt_downsweep.items_per_thread,
        alt_downsweep.radix_bits,
        alt_downsweep.load_modifier};

      const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        19,
        __dominant_size(),
        single_tile_radix_bits,
        BLOCK_LOAD_DIRECT,
        LOAD_LDG,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return radix_sort_policy{
        use_onesweep,
        onesweep_radix_bits,
        histogram,
        exclusive_sum,
        onesweep,
        scan,
        downsweep,
        alt_downsweep,
        upsweep,
        alt_upsweep,
        single_tile,
        segmented,
        alt_segmented};
    }

    // SM50
    const int primary_radix_bits     = (key_size > 1) ? 7 : 5; // 3.5B 32b keys/s, 1.92B 32b pairs/s (TitanX)
    const int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
    const int segmented_radix_bits   = (key_size > 1) ? 6 : 5; // 3.1B 32b segmented keys/s (TitanX)
    const bool use_onesweep          = false;
    const int onesweep_radix_bits    = 8;

    const auto histogram = radix_sort_histogram_policy{256, 8, __scale_num_parts(1, key_size), onesweep_radix_bits};

    const auto exclusive_sum = radix_sort_exclusive_sum_policy{256, onesweep_radix_bits};

    const auto onesweep = make_reg_scaled_radix_sort_onesweep_policy(
      256,
      21,
      __dominant_size(),
      1,
      onesweep_radix_bits,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      RADIX_SORT_STORE_DIRECT);

    const auto scan = make_mem_scaled_scan_policy(
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
      primary_radix_bits,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_BASIC,
      BLOCK_SCAN_WARP_SCANS);

    const auto alt_downsweep = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      16,
      __dominant_size(),
      primary_radix_bits - 1,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_RAKING_MEMOIZE);

    const auto upsweep = radix_sort_upsweep_policy{
      downsweep.block_threads, downsweep.items_per_thread, downsweep.radix_bits, downsweep.load_modifier};

    const auto alt_upsweep = radix_sort_upsweep_policy{
      alt_downsweep.block_threads,
      alt_downsweep.items_per_thread,
      alt_downsweep.radix_bits,
      alt_downsweep.load_modifier};

    const auto single_tile = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      19,
      __dominant_size(),
      single_tile_radix_bits,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
      192,
      31,
      __dominant_size(),
      segmented_radix_bits,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      11,
      __dominant_size(),
      segmented_radix_bits - 1,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    return radix_sort_policy{
      use_onesweep,
      onesweep_radix_bits,
      histogram,
      exclusive_sum,
      onesweep,
      scan,
      downsweep,
      alt_downsweep,
      upsweep,
      alt_upsweep,
      single_tile,
      segmented,
      alt_segmented};
  }
};

template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(cuda::arch_id arch) const -> radix_sort_policy
  {
    constexpr auto policies = policy_selector{
      int{sizeof(KeyT)},
      ::cuda::std::is_same_v<ValueT, NullType> ? 0 : int{sizeof(ValueT)},
      int{sizeof(OffsetT)},
      classify_type<KeyT>};
    return policies(arch);
  }
};
} // namespace detail::radix_sort

CUB_NAMESPACE_END
