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
struct segmented_reduce_policy
{
  reduce::agent_reduce_policy segmented_reduce;

  _CCCL_API constexpr friend bool operator==(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return lhs.segmented_reduce == rhs.segmented_reduce;
  }

  _CCCL_API constexpr friend bool operator!=(const segmented_reduce_policy& lhs, const segmented_reduce_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_reduce_policy& p)
  {
    return os << "segmented_reduce_policy { .segmented_reduce = " << p.segmented_reduce << " }";
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
    return segmented_reduce_policy{reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(arch).reduce};
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

template <typename PolicyHub>
struct policy_selector_from_hub
{
  // this is only called in device code, so we can ignore the arch parameter
  _CCCL_DEVICE_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> segmented_reduce_policy
  {
    using p = typename PolicyHub::MaxPolicy::ActivePolicy::SegmentedReducePolicy;
    return segmented_reduce_policy{{
      p::BLOCK_THREADS,
      p::ITEMS_PER_THREAD,
      p::VECTOR_LOAD_LENGTH,
      p::BLOCK_ALGORITHM,
      p::LOAD_MODIFIER,
    }};
  }
};
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
