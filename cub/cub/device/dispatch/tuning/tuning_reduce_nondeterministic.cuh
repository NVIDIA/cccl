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

#include <cuda/__device/compute_capability.h>

CUB_NAMESPACE_BEGIN
namespace detail::reduce_nondeterministic
{
struct reduce_nondeterministic_policy
{
  reduce::agent_reduce_policy reduce;

  _CCCL_API constexpr friend bool
  operator==(const reduce_nondeterministic_policy& lhs, const reduce_nondeterministic_policy& rhs)
  {
    return lhs.reduce == rhs.reduce;
  }

  _CCCL_API constexpr friend bool
  operator!=(const reduce_nondeterministic_policy& lhs, const reduce_nondeterministic_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const reduce_nondeterministic_policy& p)
  {
    return os << "reduce_nondeterministic_policy { .reduce = " << p.reduce << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept reduce_nondeterministic_policy_selector = policy_selector<T, reduce_nondeterministic_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> reduce_nondeterministic_policy
  {
    auto policy            = reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(cc).reduce;
    policy.block_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    return {policy};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(reduce_nondeterministic_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> reduce_nondeterministic_policy
  {
    constexpr auto policies =
      policy_selector{classify_type<AccumT>, classify_op<ReductionOpT>, int{sizeof(OffsetT)}, int{sizeof(AccumT)}};
    return policies(cc);
  }
};
} // namespace detail::reduce_nondeterministic

CUB_NAMESPACE_END
