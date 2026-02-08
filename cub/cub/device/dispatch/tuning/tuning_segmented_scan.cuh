// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/void_t.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename PolicyT, typename = void, typename = void>
struct segmented_scan_policy_wrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION segmented_scan_policy_wrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct segmented_scan_policy_wrapper<
  StaticPolicyT,
  ::cuda::std::void_t<decltype(StaticPolicyT::segmented_scan_policy_t::load_modifier)>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION segmented_scan_policy_wrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr auto SegmentedScan()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::segmented_scan_policy_t());
  }

  CUB_RUNTIME_FUNCTION static constexpr CacheLoadModifier LoadModifier()
  {
    return StaticPolicyT::segmented_scan_policy_t::load_modifier;
  }

  CUB_RUNTIME_FUNCTION constexpr void CheckLoadModifier()
  {
    static_assert(LoadModifier() != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION segmented_scan_policy_wrapper<PolicyT> make_segmented_scan_policy_wrapper(PolicyT policy)
{
  return segmented_scan_policy_wrapper<PolicyT>{policy};
}

template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_hub
{
private:
  static constexpr bool large_values = sizeof(AccumT) > 128;

public:
  // For large values, use timesliced loads/stores to fit shared memory.
  static constexpr BlockLoadAlgorithm scan_transposed_load =
    large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm scan_transposed_store =
    large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

  struct default_policy
  {
    using segmented_scan_policy_t = agent_segmented_scan_policy_t<
      128,
      15,
      AccumT,
      scan_transposed_load,
      LOAD_DEFAULT,
      scan_transposed_store,
      BLOCK_SCAN_WARP_SCANS>;
  };

  struct policy_500
      : default_policy
      , ChainedPolicy<500, policy_500, policy_500>
  {};

  using MaxPolicy = policy_500;
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
