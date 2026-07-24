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

#include <cuda/__cmath/pow2.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/std/__host_stdlib/ostream>
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

  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return lhs.items_per_thread == rhs.items_per_thread && lhs.load_algorithm == rhs.load_algorithm
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool operator!=(const epilogue_policy& lhs, const epilogue_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const epilogue_policy& p)
  {
    return os
        << "epilogue_policy { .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .store_algorithm = " << p.store_algorithm << ", .scan_algorithm = " << p.scan_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
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

  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const worker_policy& lhs, const worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.store_algorithm == rhs.store_algorithm
        && lhs.epilogue == rhs.epilogue;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool operator!=(const worker_policy& lhs, const worker_policy& rhs)
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
#endif // _CCCL_HOSTED()
};

//! Sub-policy for the baseline backend's multiple-blocks-per-segment worker path, used for segments too large for a
//! single worker block.
struct multi_worker_policy
{
  int threads_per_block; //!< Number of threads in a CUDA block.
  int items_per_thread; //!< Keys each thread loads/processes per tile.

  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool operator!=(const multi_worker_policy& lhs, const multi_worker_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
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

  _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const baseline_topk_policy& lhs, const baseline_topk_policy& rhs)
  {
    return lhs.worker_per_segment_policies == rhs.worker_per_segment_policies
        && lhs.multi_worker_per_segment_policy == rhs.multi_worker_per_segment_policy;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool
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

// Default baseline sub-policy. Tuning is currently CC-independent.
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

//! Execution shape for the thread-block-cluster backend of @ref DeviceBatchedTopK. The dispatch picks the number of
//! cluster blocks and the dynamic shared-memory block_tile capacity at runtime (occupancy / wave-aware), so this policy
//! mostly carries per-block tuning knobs; the two trailing `max_*` fields are optional launch-geometry caps that bound
//! that runtime choice.
struct cluster_topk_policy
{
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

  // Launch-geometry caps that bound the otherwise heuristic / hardware-derived cluster width and resident shared-memory
  // footprint. Both default to 0 (= unrestricted) and are deliberately not auto-tuned. Beyond deterministically
  // steering tests onto the streaming / cluster paths at a small footprint, they let a caller trade top-k throughput
  // for resources it wants to leave free -- e.g. capping resident slots to fit a shared-memory carveout reserved for a
  // concurrently running kernel, or narrowing the cluster width to co-schedule other work. The algorithm stays correct
  // at any cap: a segment that no longer fits resident simply streams the remainder from global memory.
  int max_blocks_per_cluster; //!< Upper bound on the launched cluster width (CTAs per segment); 0 = unrestricted (the
                              //!< hardware cluster-width ceiling, queried from the runtime). Non-zero is additionally
                              //!< clamped to that same ceiling. A cap narrower than a segment needs pushes it into the
                              //!< streaming fallback (cap 1 -> single-CTA streaming).
  int max_chunk_slots_per_block; //!< Upper bound on resident chunk slots per block; 0 = unrestricted (the full
                                 //!< shared-memory budget: the hardware opt-in budget). A smaller cap shrinks each
                                 //!< CTA's resident capacity (and thus its dynamic shared-memory request), so a smaller
                                 //!< segment overflows into the streaming path.

  // Equality/streaming make this a regular type (required by the `policy_selector` concept / `dispatch_compute_cap`).
  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const cluster_topk_policy& lhs, const cluster_topk_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.min_blocks_per_sm == rhs.min_blocks_per_sm
        && lhs.min_chunks_per_block == rhs.min_chunks_per_block && lhs.chunk_bytes == rhs.chunk_bytes
        && lhs.load_align_bytes == rhs.load_align_bytes && lhs.pipeline_stages == rhs.pipeline_stages
        && lhs.single_block_max_seg_size == rhs.single_block_max_seg_size && lhs.bits_per_pass == rhs.bits_per_pass
        && lhs.histogram_items_per_thread == rhs.histogram_items_per_thread
        && lhs.tie_break_items_per_thread == rhs.tie_break_items_per_thread
        && lhs.copy_items_per_thread == rhs.copy_items_per_thread
        && lhs.max_blocks_per_cluster == rhs.max_blocks_per_cluster
        && lhs.max_chunk_slots_per_block == rhs.max_chunk_slots_per_block;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool operator!=(const cluster_topk_policy& lhs, const cluster_topk_policy& rhs)
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
        << p.tie_break_items_per_thread << ", .copy_items_per_thread = " << p.copy_items_per_thread
        << ", .max_blocks_per_cluster = " << p.max_blocks_per_cluster
        << ", .max_chunk_slots_per_block = " << p.max_chunk_slots_per_block << " }";
  }
#endif // _CCCL_HOSTED()
};

// Default cluster sub-policy. Tuning is currently CC-independent.
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
    /*copy_items_per_thread=*/8,
    /*max_blocks_per_cluster=*/0,
    /*max_chunk_slots_per_block=*/0};
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

// -----------------------------------------------------------------------------
// Backend selection
// -----------------------------------------------------------------------------
//! Backend algorithms for @ref DeviceBatchedTopK. Both backends are launched through a single kernel symbol; which one
//! runs is decided per architecture by `policy_selector` below, whose result also drives the device-side agent
//! selection (via `current_policy`).
enum class topk_algorithm
{
  baseline, //!< worker-per-segment backend (single thread block per segment)
  cluster, //!< thread-block-cluster backend (SM 9.0+)
  unsupported //!< no backend can serve the request on the target architecture; dispatch returns cudaErrorNotSupported
};

#if _CCCL_HOSTED()
[[nodiscard]] inline ::std::ostream& operator<<(::std::ostream& os, topk_algorithm backend)
{
  switch (backend)
  {
    case topk_algorithm::baseline:
      return os << "baseline";
    case topk_algorithm::cluster:
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
  topk_algorithm backend; //!< Backend the dispatch selected, i.e. the kernel arm that runs.
  baseline_topk_policy baseline; //!< Sub-policy used when @p backend is @p topk_algorithm::baseline.
  cluster_topk_policy cluster; //!< Sub-policy used when @p backend is @p topk_algorithm::cluster.

  _CCCL_HOST_DEVICE_API friend constexpr bool operator==(const topk_policy& lhs, const topk_policy& rhs)
  {
    return lhs.backend == rhs.backend && lhs.baseline == rhs.baseline && lhs.cluster == rhs.cluster;
  }

  _CCCL_HOST_DEVICE_API friend constexpr bool operator!=(const topk_policy& lhs, const topk_policy& rhs)
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

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept topk_policy_selector = policy_selector<T, topk_policy>;
#endif // _CCCL_HAS_CONCEPTS()

// Crossover knobs (TODO: tune via SM100 benchmarks).
//! Clusters require SM 9.0+.
inline constexpr int cluster_min_cc_major = 9;
//! Smallest statically-known maximum segment size at which the cluster backend starts to win (measured on B200). This
//! is the backend crossover threshold and is intentionally part of the selector -- not a tunable policy field -- so
//! that tuning the cluster policy (e.g. its single-CTA threshold) does not silently shift which backend is chosen.
inline constexpr ::cuda::std::int64_t cluster_beneficial_min_segment_size = 8 * 1024;

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool cluster_capable([[maybe_unused]] ::cuda::compute_capability cc)
{
#if _CCCL_HAS_DYNAMIC_CLUSTER_LAUNCH()
  return cc >= ::cuda::compute_capability{cluster_min_cc_major, 0};
#else // ^^^ dynamic cluster launches enabled ^^^ / vvv dynamic cluster launches disabled vvv
  // The cluster backend launches with a runtime cluster width, which CCCL_DISABLE_DYNAMIC_CLUSTER_LAUNCH compiles out;
  // reporting no architecture as cluster-capable makes the selector fall back to baseline (or report unsupported).
  return false;
#endif // _CCCL_HAS_DYNAMIC_CLUSTER_LAUNCH()
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
