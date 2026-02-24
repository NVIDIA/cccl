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

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN
namespace detail::segmented_reduce
{
// Runtime representation of a warp-level reduce agent policy (small/medium segments)
struct agent_warp_reduce_policy
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

  _CCCL_API constexpr friend bool operator==(const agent_warp_reduce_policy& lhs, const agent_warp_reduce_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.warp_threads == rhs.warp_threads
        && lhs.items_per_thread == rhs.items_per_thread && lhs.vector_load_length == rhs.vector_load_length
        && lhs.load_modifier == rhs.load_modifier;
  }

  _CCCL_API constexpr friend bool operator!=(const agent_warp_reduce_policy& lhs, const agent_warp_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const agent_warp_reduce_policy& p)
  {
    return os << "agent_warp_reduce_policy { .block_threads = " << p.block_threads
              << ", .warp_threads = " << p.warp_threads << ", .items_per_thread = " << p.items_per_thread
              << ", .vector_load_length = " << p.vector_load_length << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct segmented_reduce_policy
{
  reduce::agent_reduce_policy segmented_reduce;
  agent_warp_reduce_policy small_reduce;
  agent_warp_reduce_policy medium_reduce;

  _CCCL_API constexpr friend bool operator==(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return lhs.segmented_reduce == rhs.segmented_reduce && lhs.small_reduce == rhs.small_reduce
        && lhs.medium_reduce == rhs.medium_reduce;
  }

  _CCCL_API constexpr friend bool operator!=(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_reduce_policy& p)
  {
    return os << "segmented_reduce_policy { .segmented_reduce = " << p.segmented_reduce
              << ", .small_reduce = " << p.small_reduce << ", .medium_reduce = " << p.medium_reduce << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_reduce_policy_selector = policy_selector<T, segmented_reduce_policy>;
#endif // _CCCL_HAS_CONCEPTS()

template <typename PolicyT, typename = void>
struct FixedSizeSegmentedReducePolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE FixedSizeSegmentedReducePolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct FixedSizeSegmentedReducePolicyWrapper<StaticPolicyT,
                                             ::cuda::std::void_t<typename StaticPolicyT::ReducePolicy,
                                                                 typename StaticPolicyT::SmallReducePolicy,
                                                                 typename StaticPolicyT::MediumReducePolicy>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE FixedSizeSegmentedReducePolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(Reduce)
  CUB_DEFINE_SUB_POLICY_GETTER(SmallReduce)
  CUB_DEFINE_SUB_POLICY_GETTER(MediumReduce)

  _CCCL_HOST_DEVICE static constexpr int SmallReduceItemsPerTile()
  {
    return StaticPolicyT::SmallReducePolicy::ITEMS_PER_TILE;
  }

  _CCCL_HOST_DEVICE static constexpr int MediumReduceItemsPerTile()
  {
    return StaticPolicyT::MediumReducePolicy::ITEMS_PER_TILE;
  }

  _CCCL_HOST_DEVICE static constexpr int SmallReduceSegmentsPerBlock()
  {
    return StaticPolicyT::SmallReducePolicy::SEGMENTS_PER_BLOCK;
  }

  _CCCL_HOST_DEVICE static constexpr int MediumReduceSegmentsPerBlock()
  {
    return StaticPolicyT::MediumReducePolicy::SEGMENTS_PER_BLOCK;
  }

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"ReducePolicy">()       = Reduce().EncodedPolicy(),
                  key<"SmallReducePolicy">()  = SmallReduce().EncodedPolicy(),
                  key<"MediumReducePolicy">() = MediumReduce().EncodedPolicy()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE FixedSizeSegmentedReducePolicyWrapper<PolicyT>
MakeFixedSizeSegmentedReducePolicyWrapper(PolicyT policy)
{
  return FixedSizeSegmentedReducePolicyWrapper<PolicyT>{policy};
}

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

struct policy_selector
{
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> segmented_reduce_policy
  {
    constexpr int threads_per_block  = 256;
    constexpr int items_per_thread   = 16;
    constexpr int items_per_vec_load = 4;

    auto [scaled_items, scaled_threads] = scale_mem_bound(threads_per_block, items_per_thread, accum_size);
    const auto base                     = reduce::agent_reduce_policy{
      scaled_threads, scaled_items, items_per_vec_load, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};

    return segmented_reduce_policy{
      base,
      agent_warp_reduce_policy{base.block_threads, 1, items_per_thread, items_per_vec_load, base.load_modifier},
      agent_warp_reduce_policy{base.block_threads, 32, items_per_thread, items_per_vec_load, base.load_modifier}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_reduce_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> segmented_reduce_policy
  {
    using fs        = typename policy_hub<AccumT, OffsetT, ReductionOpT>::MaxPolicy;
    using rp        = typename fs::ReducePolicy;
    using sp        = typename fs::SmallReducePolicy;
    using mp        = typename fs::MediumReducePolicy;
    const auto base = reduce::agent_reduce_policy{
      rp::BLOCK_THREADS, rp::ITEMS_PER_THREAD, rp::VECTOR_LOAD_LENGTH, rp::BLOCK_ALGORITHM, rp::LOAD_MODIFIER};
    return segmented_reduce_policy{
      base,
      agent_warp_reduce_policy{
        base.block_threads, sp::WARP_THREADS, sp::ITEMS_PER_THREAD, sp::VECTOR_LOAD_LENGTH, sp::LOAD_MODIFIER},
      agent_warp_reduce_policy{
        base.block_threads, mp::WARP_THREADS, mp::ITEMS_PER_THREAD, mp::VECTOR_LOAD_LENGTH, mp::LOAD_MODIFIER}};
  }
};
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
