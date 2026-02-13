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

struct policy_selector
{
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> segmented_reduce_policy
  {
    // for now the segmented reduction uses the same tuning values as the normal reduction
    const auto base = reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(arch).reduce;
    return segmented_reduce_policy{
      base,
      agent_warp_reduce_policy{base.block_threads, 1, 16, base.vector_load_length, base.load_modifier},
      agent_warp_reduce_policy{base.block_threads, 32, 16, base.vector_load_length, base.load_modifier}};
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
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
