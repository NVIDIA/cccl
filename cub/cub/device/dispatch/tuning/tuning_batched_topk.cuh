// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/pow2.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail::batched_topk
{
//! Sub-policy for the compaction epilogue shared by the baseline @ref DeviceBatchedTopK workers: it scans the radix
//! histogram and writes out the selected keys.
struct epilogue_policy
{
  int items_per_thread; //!< Keys each thread loads/stores per tile in the epilogue.
  BlockLoadAlgorithm load_algorithm; //!< Block load algorithm used to read keys back in the epilogue.
  BlockStoreAlgorithm store_algorithm; //!< Block store algorithm used to write the selected keys.
  BlockScanAlgorithm scan_algorithm; //!< Block scan algorithm used for the histogram prefix sum.

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return lhs.items_per_thread == rhs.items_per_thread && lhs.load_algorithm == rhs.load_algorithm
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const epilogue_policy& p)
  {
    return os
        << "epilogue_policy { .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .store_algorithm = " << p.store_algorithm << ", .scan_algorithm = " << p.scan_algorithm << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

//! Per-segment worker sub-policy for the baseline backend: one thread block cooperatively computes the top-k of a
//! single segment. @ref baseline_topk_policy holds several of these, ordered by decreasing tile size.
struct worker_policy
{
  int threads_per_block; //!< Number of threads in a CUDA block.
  int items_per_thread; //!< Keys each thread loads/processes per tile (with `threads_per_block` sets the tile size).
  BlockLoadAlgorithm load_algorithm; //!< Block load algorithm used to read the segment's keys.
  BlockStoreAlgorithm store_algorithm; //!< Block store algorithm used to write the selected keys.

  epilogue_policy epilogue; //!< Sub-policy for the compaction epilogue.

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const worker_policy& lhs, const worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.store_algorithm == rhs.store_algorithm
        && lhs.epilogue == rhs.epilogue;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const worker_policy& lhs, const worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const worker_policy& p)
  {
    return os << "worker_policy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
              << ", .store_algorithm = " << p.store_algorithm << ", .epilogue = " << p.epilogue << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

//! Sub-policy for the baseline backend's multiple-blocks-per-segment worker path, used for segments too large for a
//! single worker block.
struct multi_worker_policy
{
  int threads_per_block; //!< Number of threads in a CUDA block.
  int items_per_thread; //!< Keys each thread loads/processes per tile.

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const multi_worker_policy& p)
  {
    return os << "multi_worker_policy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

//! Sub-policy for the baseline (worker-per-segment) backend of @ref DeviceBatchedTopK.
struct baseline_topk_policy
{
  //! Per-segment worker policies ordered by decreasing tile size. At compile time the smallest policy whose tile size
  //! still covers the upper bound of the segment size is selected.
  ::cuda::std::array<worker_policy, 6> worker_per_segment_policies;
  multi_worker_policy multi_worker_per_segment_policy; //!< Worker policy for segments too large for a single block.

  _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const baseline_topk_policy& lhs, const baseline_topk_policy& rhs)
  {
    return lhs.worker_per_segment_policies == rhs.worker_per_segment_policies
        && lhs.multi_worker_per_segment_policy == rhs.multi_worker_per_segment_policy;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const baseline_topk_policy& lhs, const baseline_topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const baseline_topk_policy& p)
  {
    os << "baseline_topk_policy { .worker_per_segment_policies = { ";
    for (::cuda::std::size_t i = 0; i < p.worker_per_segment_policies.size(); ++i)
    {
      if (i != 0)
      {
        os << ", ";
      }
      os << p.worker_per_segment_policies[i];
    }
    return os << " }, .multi_worker_per_segment_policy = " << p.multi_worker_per_segment_policy << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept baseline_topk_policy_selector = policy_selector<T, baseline_topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

// Default baseline sub-policy. A free function so both `baseline_policy_selector` and the combined `policy_selector`
// can build it inline. Tuning is currently CC-independent.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_baseline_policy() -> baseline_topk_policy
{
  constexpr auto load_alg  = BLOCK_LOAD_WARP_TRANSPOSE;
  constexpr auto store_alg = BLOCK_STORE_WARP_TRANSPOSE;
  constexpr auto scan_alg  = BLOCK_SCAN_WARP_SCANS;
  constexpr auto epilogue  = epilogue_policy{16, load_alg, store_alg, scan_alg};
  return baseline_topk_policy{
    {{
      worker_policy{256, 64, load_alg, store_alg, epilogue},
      worker_policy{256, 32, load_alg, store_alg, epilogue},
      worker_policy{256, 16, load_alg, store_alg, epilogue},
      worker_policy{256, 8, load_alg, store_alg, epilogue},
      worker_policy{256, 4, load_alg, store_alg, epilogue},
      worker_policy{128, 2, load_alg, store_alg, epilogue},
    }},
    multi_worker_policy{256, 64}};
}

// Largest maximum segment size (in keys) the baseline (worker-per-segment) backend can cover: the largest worker tile
// (threads_per_block * items_per_thread) in `policy`. A larger statically-known maximum segment size makes the baseline
// backend ineligible (the selector then picks the cluster backend where supported, otherwise `unsupported`). This is
// only the tile-based necessary condition; the exact predicate `baseline_can_cover_v` also checks the agent's
// shared-memory fit (which needs the concrete agent types).
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::int64_t
baseline_max_covered_segment_size(const baseline_topk_policy& policy)
{
  ::cuda::std::int64_t max_tile_size = 0;
  for (const auto& worker : policy.worker_per_segment_policies)
  {
    const ::cuda::std::int64_t tile_size = ::cuda::std::int64_t{worker.threads_per_block} * worker.items_per_thread;
    if (tile_size > max_tile_size)
    {
      max_tile_size = tile_size;
    }
  }
  return max_tile_size;
}

struct baseline_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const
    -> baseline_topk_policy
  {
    return make_baseline_policy();
  }
};

template <typename KeyT, typename ValueT, typename SegmentSizeT, ::cuda::std::int64_t MaxK>
struct baseline_policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> baseline_topk_policy
  {
    return baseline_policy_selector{}(cc);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(baseline_topk_policy_selector<baseline_policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

//! Per-block execution shape for the thread-block-cluster backend of @ref DeviceBatchedTopK. The dispatch picks the
//! number of cluster blocks and the dynamic shared-memory block_tile capacity at runtime (occupancy / wave-aware), so
//! this policy carries only the per-block tuning knobs.
struct cluster_topk_policy
{
  // Fields grouped by kind and, across and within groups, ordered by kernel use: launch config, then load, then
  // process, then output.

  int threads_per_block; //!< Number of threads in a CUDA block.
  int min_blocks_per_sm; //!< Minimum resident blocks per SM, forwarded as the kernel launch-bounds occupancy hint.

  int min_chunks_per_block; //!< Minimum number of chunks a block must own to join a segment's effective cluster (the
                            //!< divisor mapping a segment's chunk count to its cluster width). Must be >= 1.

  int chunk_bytes; //!< Size in bytes of one block_tile chunk -- the granularity of the async-copy load pipeline.
  int load_align_bytes; //!< Load / bulk-copy alignment in bytes. Must be a power of two and >= 16
                        //!< (`detail::bulk_copy_min_align`), and `chunk_bytes` must be a multiple of it (see
                        //!< `is_valid_cluster_policy`).
  int pipeline_stages; //!< Depth of the async-copy (mbarrier) pipeline that stages chunks into shared memory.

  int single_block_max_seg_size; //!< Largest segment size, in keys, still eligible for the single-block fast path
                                 //!< (kept out of the byte-unit loading group above as it is measured in items).
  int bits_per_pass; //!< Radix digit width per pass; each pass' histogram spans `1 << bits_per_pass` buckets. Together
                     //!< with `threads_per_block` this implicitly fixes the histogram block-scan's items per thread,
                     //!< `ceil_div(1 << bits_per_pass, threads_per_block)` buckets scanned per thread.
  int histogram_items_per_thread; //!< Keys each thread accumulates per tile during the radix histogram passes.
  int tie_break_items_per_thread; //!< Keys each thread processes per tile during the final tie-break / filter phase.
  int copy_items_per_thread; //!< Keys each thread copies per tile on the select-all (k >= segment size) fast path.

  // Equality/streaming make this a regular type (required by the `policy_selector` concept / `dispatch_compute_cap`).
  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const cluster_topk_policy& lhs, const cluster_topk_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.min_blocks_per_sm == rhs.min_blocks_per_sm
        && lhs.min_chunks_per_block == rhs.min_chunks_per_block && lhs.chunk_bytes == rhs.chunk_bytes
        && lhs.load_align_bytes == rhs.load_align_bytes && lhs.pipeline_stages == rhs.pipeline_stages
        && lhs.single_block_max_seg_size == rhs.single_block_max_seg_size && lhs.bits_per_pass == rhs.bits_per_pass
        && lhs.histogram_items_per_thread == rhs.histogram_items_per_thread
        && lhs.tie_break_items_per_thread == rhs.tie_break_items_per_thread
        && lhs.copy_items_per_thread == rhs.copy_items_per_thread;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const cluster_topk_policy& lhs, const cluster_topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const cluster_topk_policy& p)
  {
    return os
        << "cluster_topk_policy { .threads_per_block = " << p.threads_per_block
        << ", .min_blocks_per_sm = " << p.min_blocks_per_sm << ", .min_chunks_per_block = " << p.min_chunks_per_block
        << ", .chunk_bytes = " << p.chunk_bytes << ", .load_align_bytes = " << p.load_align_bytes
        << ", .pipeline_stages = " << p.pipeline_stages
        << ", .single_block_max_seg_size = " << p.single_block_max_seg_size << ", .bits_per_pass = " << p.bits_per_pass
        << ", .histogram_items_per_thread = " << p.histogram_items_per_thread << ", .tie_break_items_per_thread = "
        << p.tie_break_items_per_thread << ", .copy_items_per_thread = " << p.copy_items_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept cluster_topk_policy_selector = policy_selector<T, cluster_topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

// Default cluster sub-policy. A free function so both `cluster_policy_selector` and the combined `policy_selector`
// can build it inline. Tuning is currently CC-independent.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto make_cluster_policy() -> cluster_topk_policy
{
  return cluster_topk_policy{
    /*threads_per_block=*/512,
    /*min_blocks_per_sm=*/1,
    /*min_chunks_per_block=*/1,
    /*chunk_bytes=*/16 * 1024,
    /*load_align_bytes=*/128,
    /*pipeline_stages=*/8,
    /*single_block_max_seg_size=*/8 * 1024,
    /*bits_per_pass=*/11,
    /*histogram_items_per_thread=*/8,
    /*tie_break_items_per_thread=*/8,
    /*copy_items_per_thread=*/8};
}

// Hard constraints on the block_tile byte geometry. The aligned bulk-copy (TMA) load path addresses gmem/smem in
// `load_align_bytes`-sized, aligned units, so `load_align_bytes` must be a power of two and at least
// `bulk_copy_min_align` (16 B), and `chunk_bytes` must be a whole number of those units. `launch_cluster_arm`
// static-asserts this before instantiating the agent.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto is_valid_cluster_policy(cluster_topk_policy policy) -> bool
{
  return policy.load_align_bytes >= bulk_copy_min_align && ::cuda::is_power_of_two(policy.load_align_bytes)
      && policy.chunk_bytes % policy.load_align_bytes == 0;
}

static_assert(is_valid_cluster_policy(make_cluster_policy()));

// Tuned cluster sub-policies for SM 10.x pairs (key + index) requests under a deterministic result-set requirement,
// measured on B200 (EVO search over `cub.bench.segmented_topk.variable.indexed.cluster` with F32 keys and I32
// indices, deterministic gpu-to-gpu + prefer-larger-index). Buckets are keyed by the request's statically-known
// upper bounds on segment size and k; bounds outside the measured grid fall back to the default policy. Grid
// diagonal cells (k == max segment size) are intentionally not encoded: the benchmark there only exercises the
// select-all copy fast path (every segment has exactly k elements), leaving the selection-path knobs unmeasured.
// Such requests fall through to the nearest larger measured bucket (or the default).
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
make_sm100_pairs_cluster_policy(::cuda::std::int64_t static_max_segment_size, ::cuda::std::int64_t max_k)
  -> cluster_topk_policy
{
  if (static_max_segment_size <= 1024 && max_k <= 512)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/352,
      /*min_blocks_per_sm=*/1,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/27 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/13,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/8,
      /*histogram_items_per_thread=*/6,
      /*tie_break_items_per_thread=*/8,
      /*copy_items_per_thread=*/9};
  }
  if (static_max_segment_size <= 2048 && max_k <= 512)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/352,
      /*min_blocks_per_sm=*/1,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/30 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/14,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/8,
      /*histogram_items_per_thread=*/12,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/14};
  }
  if (static_max_segment_size <= 2048 && max_k <= 1024)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/416,
      /*min_blocks_per_sm=*/1,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/23 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/4,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/8,
      /*histogram_items_per_thread=*/3,
      /*tie_break_items_per_thread=*/1,
      /*copy_items_per_thread=*/4};
  }
  if (static_max_segment_size <= 4096 && max_k <= 512)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/416,
      /*min_blocks_per_sm=*/1,
      /*min_chunks_per_block=*/2,
      /*chunk_bytes=*/15 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/9,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/8,
      /*histogram_items_per_thread=*/3,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/14};
  }
  if (static_max_segment_size <= 4096 && max_k <= 1024)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/512,
      /*min_blocks_per_sm=*/2,
      /*min_chunks_per_block=*/2,
      /*chunk_bytes=*/16 * 1024,
      /*load_align_bytes=*/32,
      /*pipeline_stages=*/7,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/8,
      /*histogram_items_per_thread=*/2,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/24};
  }
  if (static_max_segment_size <= 4096 && max_k <= 2048)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/512,
      /*min_blocks_per_sm=*/1,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/27 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/10,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/9,
      /*histogram_items_per_thread=*/2,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/12};
  }
  if (static_max_segment_size <= 8192 && max_k <= 512)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/512,
      /*min_blocks_per_sm=*/2,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/23 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/8,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/11,
      /*histogram_items_per_thread=*/3,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/16};
  }
  if (static_max_segment_size <= 8192 && max_k <= 1024)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/512,
      /*min_blocks_per_sm=*/2,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/24 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/11,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/9,
      /*histogram_items_per_thread=*/2,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/7};
  }
  if (static_max_segment_size <= 8192 && max_k <= 2048)
  {
    return cluster_topk_policy{
      /*threads_per_block=*/512,
      /*min_blocks_per_sm=*/2,
      /*min_chunks_per_block=*/1,
      /*chunk_bytes=*/29 * 1024,
      /*load_align_bytes=*/16,
      /*pipeline_stages=*/7,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/11,
      /*histogram_items_per_thread=*/2,
      /*tie_break_items_per_thread=*/2,
      /*copy_items_per_thread=*/6};
  }
  return make_cluster_policy();
}

static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(1024, 512)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(2048, 512)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(2048, 1024)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(4096, 512)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(4096, 1024)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(4096, 2048)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(8192, 512)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(8192, 1024)));
static_assert(is_valid_cluster_policy(make_sm100_pairs_cluster_policy(8192, 2048)));

// Default selector for cluster-capable architectures (SM 9.0+). The tuning is currently identical across CCs.
struct cluster_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> cluster_topk_policy
  {
    return make_cluster_policy();
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(cluster_topk_policy_selector<cluster_policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------
//! Backend algorithms for @ref DeviceBatchedTopK. Both backends are launched through a single kernel symbol; which one
//! runs is decided per architecture by `policy_selector` below, whose result also drives the device-side agent
//! selection (via `current_policy`).
enum class topk_backend
{
  baseline, //!< worker-per-segment backend (single thread block per segment)
  cluster, //!< thread-block-cluster backend (SM 9.0+)
  unsupported //!< no backend can serve the request on the target architecture; dispatch returns cudaErrorNotSupported
};

#if _CCCL_HOSTED()
[[nodiscard]] inline ::std::ostream& operator<<(::std::ostream& os, topk_backend backend)
{
  switch (backend)
  {
    case topk_backend::baseline:
      return os << "baseline";
    case topk_backend::cluster:
      return os << "cluster";
    default:
      return os << "unsupported";
  }
}
#endif // _CCCL_HOSTED()

//! The tuning policy for all backends of @ref DeviceBatchedTopK. It carries the selected backend plus both backends'
//! sub-policies; the kernel instantiates only the arm named by @p backend (chosen device-side via `current_policy`).
//!
//! This is a regular type: `detail::dispatch_compute_cap` (and the `policy_selector` concept) require the selector's
//! result to be `::cuda::std::regular`, hence the equality/streaming operators below.
struct topk_policy
{
  topk_backend backend; //!< Backend the dispatch selected, i.e. the kernel arm that runs.
  baseline_topk_policy baseline; //!< Sub-policy used when @p backend is @p topk_backend::baseline.
  cluster_topk_policy cluster; //!< Sub-policy used when @p backend is @p topk_backend::cluster.

  _CCCL_HOST_DEVICE_API constexpr friend bool operator==(const topk_policy& lhs, const topk_policy& rhs)
  {
    return lhs.backend == rhs.backend && lhs.baseline == rhs.baseline && lhs.cluster == rhs.cluster;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool operator!=(const topk_policy& lhs, const topk_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const topk_policy& p)
  {
    return os << "topk_policy { .backend = " << p.backend << ", .baseline = " << p.baseline
              << ", .cluster = " << p.cluster << " }";
  }
#endif // _CCCL_HOSTED()
};

// Sentinel meaning "no backend-selecting tuning override was provided", so the dispatch builds its automatic selector.
// Used as the default for `dispatch`'s `PolicySelectorOverride` and as the not-found result of the public
// API's `topk_policy` tuning query (a real empty type -- `void` cannot be a query default).
struct no_override
{};

// Crossover knobs (TODO: tune via SM100 benchmarks).
//! Clusters require SM 9.0+.
inline constexpr int cluster_min_cc_major = 9;
//! The cluster backend only consistently wins at/above SM 10.0 (measured on B200), and only for large segments.
inline constexpr int cluster_beneficial_min_cc_major = 10;
//! Smallest statically-known maximum segment size at which the cluster backend starts to win (measured on B200). This
//! is the backend crossover threshold and is intentionally part of the selector -- not a tunable policy field -- so
//! that tuning the cluster policy (e.g. its single-CTA threshold) does not silently shift which backend is chosen.
inline constexpr ::cuda::std::int64_t cluster_beneficial_min_segment_size = 8 * 1024;

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool cluster_capable(::cuda::compute_capability cc)
{
  return cc >= ::cuda::compute_capability{cluster_min_cc_major, 0};
}

// How the backend is chosen. `automatic` applies the determinism/arch/size rules below; the `force_*` modes let a
// caller pin a specific backend (e.g. a tuning override that forces the cluster backend while still launching the
// single kernel symbol).
enum class backend_mode
{
  automatic,
  force_baseline,
  force_cluster,
};

// Field-based backend selector (like DeviceScan's / DeviceTransform's `policy_selector`): one selector that builds both
// sub-policies inline and makes the backend decision from plain value fields (keeping it a regular value type).
// `baseline_can_cover` is supplied by the dispatch, which alone knows the concrete agent types needed for the
// shared-memory fit, so this selector stays free of the agent-type-dependent `find_smallest_covering_policy` machinery.
struct policy_selector
{
  ::cuda::std::int64_t static_max_segment_size;
  ::cuda::std::int64_t max_k;
  int key_size;
  bool keys_only;
  ::cuda::execution::determinism::__determinism_t determinism;
  ::cuda::execution::tie_break::__tie_break_t tie_break;
  bool baseline_can_cover;
  backend_mode mode;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    // A deterministic result set (or a concrete tie-break preference) can only be honored by the cluster backend.
    const bool deterministic = (determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed)
                            || (tie_break != ::cuda::execution::tie_break::__tie_break_t::__unspecified);

    const auto baseline = make_baseline_policy();

    // SM 10.x cluster tunings so far cover deterministic pairs requests with 4-byte keys (measured on B200 under a
    // prefer-larger-index tie-break; applied to both tie-break directions on the expectation of near-symmetry).
    // Other request shapes keep the default policy until their measurements land.
    const bool has_sm100_tuning = deterministic && !keys_only && key_size == 4
                               && cc >= ::cuda::compute_capability{10, 0} && cc < ::cuda::compute_capability{11, 0};
    const auto cluster =
      has_sm100_tuning ? make_sm100_pairs_cluster_policy(static_max_segment_size, max_k) : make_cluster_policy();

    topk_backend backend = topk_backend::unsupported;
    if (mode == backend_mode::force_cluster)
    {
      backend = cluster_capable(cc) ? topk_backend::cluster : topk_backend::unsupported;
    }
    else if (mode == backend_mode::force_baseline)
    {
      // The baseline backend cannot honor a deterministic result set / concrete tie-break preference, nor cover an
      // oversize segment; reject (map to `unsupported`) in those cases rather than pinning a backend that cannot serve
      // the request -- matching the hard constraints `selector_override_adaptor` enforces.
      backend = (baseline_can_cover && !deterministic) ? topk_backend::baseline : topk_backend::unsupported;
    }
    else if (deterministic)
    {
      // Deterministic -> cluster (arch permitting), independent of the max segment size.
      backend = cluster_capable(cc) ? topk_backend::cluster : topk_backend::unsupported;
    }
    else if (!baseline_can_cover)
    {
      // Oversize for the baseline backend: it must never be selected (its `find_smallest_covering_policy` would fail).
      backend = cluster_capable(cc) ? topk_backend::cluster : topk_backend::unsupported;
    }
    else
    {
      // Baseline can cover: prefer the cluster backend only where it is beneficial, otherwise use the baseline. The
      // size crossover is a fixed selector constant (not read from the tunable cluster policy), so tuning the cluster
      // policy never shifts the backend choice.
      const bool beneficial = cc >= ::cuda::compute_capability{cluster_beneficial_min_cc_major, 0}
                           && static_max_segment_size >= cluster_beneficial_min_segment_size;
      backend = (cluster_capable(cc) && beneficial) ? topk_backend::cluster : topk_backend::baseline;
    }
    return topk_policy{backend, baseline, cluster};
  }
};

// Stateless selector built purely from the compile-time request facts; default-constructs the field-based
// `policy_selector` and delegates. This is the type threaded into the dispatch and kernel: `current_policy` /
// `dispatch_compute_cap` default-construct it, so the behavior must live in the type. The facts are re-exposed as
// static members so the tuning-override adaptor can borrow them.
template <class KeyT,
          class ValueT,
          class OffsetT,
          ::cuda::std::int64_t MaxK,
          ::cuda::std::int64_t StaticMaxSegSize,
          ::cuda::execution::determinism::__determinism_t Determinism,
          ::cuda::execution::tie_break::__tie_break_t TieBreak,
          bool BaselineCanCover,
          backend_mode Mode = backend_mode::automatic>
struct policy_selector_from_types
{
  static constexpr auto determinism                             = Determinism;
  static constexpr auto tie_break                               = TieBreak;
  static constexpr bool baseline_can_cover                      = BaselineCanCover;
  static constexpr ::cuda::std::int64_t static_max_segment_size = StaticMaxSegSize;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    return policy_selector{
      StaticMaxSegSize,
      MaxK,
      int{sizeof(KeyT)},
      ::cuda::std::is_same_v<ValueT, NullType>,
      Determinism,
      TieBreak,
      BaselineCanCover,
      Mode}(cc);
  }
};

// Adapts a tune-provided selector (which only implements `operator() -> topk_policy`) to the full `PolicySelector`
// interface. The request facts (`baseline_can_cover` / `determinism` / `tie_break` / `static_max_segment_size`) are
// borrowed from the dispatch's automatic selector -- they are properties of the call, not the tuning.
// `baseline_can_cover` reflects the *default* baseline sub-policy's coverage (the adaptor lacks the concrete agent
// types needed to recompute it for an override's baseline sub-policy), so a tuning override that changes the baseline
// tile shape is still validated against the default's coverage.
//
// The override's sub-policies are forwarded verbatim and its `.backend` is honored, but validated against the chosen
// backend's hard constraints: an override selecting a backend that cannot serve the request is *rejected* (mapped to
// `unsupported`) rather than silently rerouted, surfacing the misconfiguration (compile error in strict mode, else
// cudaErrorNotSupported at runtime). Rejected cases:
//   * baseline for a deterministic / tie-break request or a segment size it cannot cover, and
//   * cluster on a pre-SM90 architecture.
template <class Override, class Default>
struct selector_override_adaptor
{
  static constexpr auto determinism                             = Default::determinism;
  static constexpr auto tie_break                               = Default::tie_break;
  static constexpr bool baseline_can_cover                      = Default::baseline_can_cover;
  static constexpr ::cuda::std::int64_t static_max_segment_size = Default::static_max_segment_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> topk_policy
  {
    const auto overridden = Override{}(cc);

    constexpr bool deterministic = (determinism != ::cuda::execution::determinism::__determinism_t::__not_guaranteed)
                                || (tie_break != ::cuda::execution::tie_break::__tie_break_t::__unspecified);

    topk_backend backend = overridden.backend;
    if (backend == topk_backend::baseline)
    {
      // The baseline backend cannot honor a deterministic / tie-break request, nor cover an oversize segment.
      if (deterministic || !baseline_can_cover)
      {
        backend = topk_backend::unsupported;
      }
    }
    else if (backend == topk_backend::cluster)
    {
      // The cluster backend requires SM 9.0+.
      if (!cluster_capable(cc))
      {
        backend = topk_backend::unsupported;
      }
    }
    return topk_policy{backend, overridden.baseline, overridden.cluster};
  }
};

// Adapts a (combined) policy selector to a plain baseline policy selector (returns just the `.baseline` sub-policy), so
// the kernel can drive `find_smallest_covering_policy` from a single `PolicySelector` template parameter.
template <class PolicySelector>
struct baseline_policy_selector_adaptor
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> baseline_topk_policy
  {
    return PolicySelector{}(cc).baseline;
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(
  detail::policy_selector<policy_selector_from_types<int,
                                                     int,
                                                     ::cuda::std::int64_t,
                                                     1024,
                                                     1024,
                                                     ::cuda::execution::determinism::__determinism_t::__not_guaranteed,
                                                     ::cuda::execution::tie_break::__tie_break_t::__unspecified,
                                                     true>,
                          topk_policy>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::batched_topk

CUB_NAMESPACE_END
