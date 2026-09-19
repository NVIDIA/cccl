// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/device/dispatch/tuning/tuning_reduce.cuh>

#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

//! The tuning policy for small/medium segments in @ref DeviceSegmentedReduce.
struct SegmentedReduceWarpReducePolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int threads_per_warp; //!< Number of threads per warp
  int items_per_thread; //!< Number of items processed per thread
  int vec_size; //!< Number of items per vectorized load
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory

  _CCCL_HOST_DEVICE_API constexpr int items_per_tile() const
  {
    return threads_per_warp * items_per_thread;
  }

  _CCCL_HOST_DEVICE_API constexpr int segments_per_block() const
  {
    return threads_per_block / threads_per_warp;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SegmentedReduceWarpReducePolicy& lhs, const SegmentedReduceWarpReducePolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.threads_per_warp == rhs.threads_per_warp
        && lhs.items_per_thread == rhs.items_per_thread && lhs.vec_size == rhs.vec_size
        && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SegmentedReduceWarpReducePolicy& lhs, const SegmentedReduceWarpReducePolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SegmentedReduceWarpReducePolicy& p)
  {
    return os << "SegmentedReduceWarpReducePolicy { .threads_per_block = " << p.threads_per_block
              << ", .threads_per_warp = " << p.threads_per_warp << ", .items_per_thread = " << p.items_per_thread
              << ", .vec_size = " << p.vec_size << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all algorithms in @ref DeviceSegmentedReduce.
struct SegmentedReducePolicy
{
  ReducePassPolicy large_reduce; //!< Policy used for large segments (one block per segment)
  SegmentedReduceWarpReducePolicy medium_reduce; //!< Policy used for medium segments (one warp per segment)
  SegmentedReduceWarpReducePolicy small_reduce; //!< Policy used for small segments (one thread per segment)

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SegmentedReducePolicy& lhs, const SegmentedReducePolicy& rhs) noexcept
  {
    return lhs.large_reduce == rhs.large_reduce && lhs.medium_reduce == rhs.medium_reduce
        && lhs.small_reduce == rhs.small_reduce;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SegmentedReducePolicy& lhs, const SegmentedReducePolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SegmentedReducePolicy& p)
  {
    return os << "SegmentedReducePolicy { .large_reduce = " << p.large_reduce
              << ", .medium_reduce = " << p.medium_reduce << ", .small_reduce = " << p.small_reduce << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::segmented_reduce
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_reduce_policy_selector = policy_selector<T, SegmentedReducePolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> SegmentedReducePolicy
  {
    constexpr int small_threads_per_warp  = 1;
    constexpr int medium_threads_per_warp = 32;

    const auto rp = reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(cc).multi_tile;

    return SegmentedReducePolicy{
      rp,
      SegmentedReduceWarpReducePolicy{
        rp.threads_per_block, medium_threads_per_warp, rp.items_per_thread, rp.vec_size, rp.load_modifier},
      SegmentedReduceWarpReducePolicy{
        rp.threads_per_block, small_threads_per_warp, rp.items_per_thread, rp.vec_size, rp.load_modifier}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_reduce_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> SegmentedReducePolicy
  {
    constexpr auto policies =
      policy_selector{classify_type<AccumT>, classify_op<ReductionOpT>, int{sizeof(OffsetT)}, int{sizeof(AccumT)}};
    return policies(cc);
  }
};

template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : detail::chained_policy<500, Policy500, Policy500>
  {
  private:
    static constexpr int items_per_vec_load = 4;

    static constexpr int small_threads_per_warp  = 1;
    static constexpr int medium_threads_per_warp = 32;

    static constexpr int nominal_4b_large_threads_per_block = 256;

    static constexpr int nominal_4b_small_items_per_thread  = 16;
    static constexpr int nominal_4b_medium_items_per_thread = 16;
    static constexpr int nominal_4b_large_items_per_thread  = 16;

  public:
    using ReducePolicy =
      agent_reduce_policy<nominal_4b_large_threads_per_block,
                          nominal_4b_large_items_per_thread,
                          AccumT,
                          items_per_vec_load,
                          cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                          cub::LOAD_LDG>;

    using SmallReducePolicy =
      agent_warp_reduce_policy<ReducePolicy::BLOCK_THREADS,
                               small_threads_per_warp,
                               nominal_4b_small_items_per_thread,
                               AccumT,
                               items_per_vec_load,
                               cub::LOAD_LDG>;

    using MediumReducePolicy =
      agent_warp_reduce_policy<ReducePolicy::BLOCK_THREADS,
                               medium_threads_per_warp,
                               nominal_4b_medium_items_per_thread,
                               AccumT,
                               items_per_vec_load,
                               cub::LOAD_LDG>;
  };

  using MaxPolicy = Policy500;
};
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
