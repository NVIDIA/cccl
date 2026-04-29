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
namespace detail::segmented_reduce
{
// for small/medium segments
struct warp_reduce_policy
{
  int block_threads;
  int warp_threads;
  int items_per_thread;
  int vector_load_length;
  CacheLoadModifier load_modifier;

  _CCCL_API constexpr int items_per_tile() const
  {
    return warp_threads * items_per_thread;
  }

  _CCCL_API constexpr int segments_per_block() const
  {
    return block_threads / warp_threads;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const warp_reduce_policy& lhs, const warp_reduce_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.warp_threads == rhs.warp_threads
        && lhs.items_per_thread == rhs.items_per_thread && lhs.vector_load_length == rhs.vector_load_length
        && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const warp_reduce_policy& lhs, const warp_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const warp_reduce_policy& p)
  {
    return os << "warp_reduce_policy { .block_threads = " << p.block_threads << ", .warp_threads = " << p.warp_threads
              << ", .items_per_thread = " << p.items_per_thread << ", .vector_load_length = " << p.vector_load_length
              << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // _CCCL_HOSTED()
};

struct segmented_reduce_policy
{
  reduce::agent_reduce_policy large_reduce;
  warp_reduce_policy small_reduce;
  warp_reduce_policy medium_reduce;

  _CCCL_API constexpr friend bool operator==(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return lhs.large_reduce == rhs.large_reduce && lhs.small_reduce == rhs.small_reduce
        && lhs.medium_reduce == rhs.medium_reduce;
  }

  _CCCL_API constexpr friend bool operator!=(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_reduce_policy& p)
  {
    return os << "segmented_reduce_policy { .large_reduce = " << p.large_reduce
              << ", .small_reduce = " << p.small_reduce << ", .medium_reduce = " << p.medium_reduce << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_reduce_policy_selector = policy_selector<T, segmented_reduce_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> segmented_reduce_policy
  {
    constexpr int small_threads_per_warp  = 1;
    constexpr int medium_threads_per_warp = 32;

    const auto rp = reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(arch).reduce;

    return segmented_reduce_policy{
      rp,
      warp_reduce_policy{
        rp.block_threads, small_threads_per_warp, rp.items_per_thread, rp.vector_load_length, rp.load_modifier},
      warp_reduce_policy{
        rp.block_threads, medium_threads_per_warp, rp.items_per_thread, rp.vector_load_length, rp.load_modifier}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_reduce_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> segmented_reduce_policy
  {
    constexpr auto policies =
      policy_selector{classify_type<AccumT>, classify_op<ReductionOpT>, int{sizeof(OffsetT)}, int{sizeof(AccumT)}};
    return policies(arch);
  }
};

template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
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
      cub::AgentReducePolicy<nominal_4b_large_threads_per_block,
                             nominal_4b_large_items_per_thread,
                             AccumT,
                             items_per_vec_load,
                             cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                             cub::LOAD_LDG>;

    using SmallReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
                                 small_threads_per_warp,
                                 nominal_4b_small_items_per_thread,
                                 AccumT,
                                 items_per_vec_load,
                                 cub::LOAD_LDG>;

    using MediumReducePolicy =
      cub::AgentWarpReducePolicy<ReducePolicy::BLOCK_THREADS,
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
