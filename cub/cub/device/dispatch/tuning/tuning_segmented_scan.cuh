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
#include <cub/device/dispatch/tuning/tuning_scan.cuh>
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
struct SegmentedScanPolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION SegmentedScanPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct SegmentedScanPolicyWrapper<StaticPolicyT,
                                  ::cuda::std::void_t<decltype(StaticPolicyT::SegmentedScanPolicyT::load_modifier)>>
    : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION SegmentedScanPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_RUNTIME_FUNCTION static constexpr auto SegmentedScan()
  {
    return cub::detail::MakePolicyWrapper(typename StaticPolicyT::SegmentedScanPolicyT());
  }

  CUB_RUNTIME_FUNCTION static constexpr CacheLoadModifier LoadModifier()
  {
    return StaticPolicyT::SegmentedScanPolicyT::load_modifier;
  }

  CUB_RUNTIME_FUNCTION constexpr void CheckLoadModifier()
  {
    static_assert(LoadModifier() != CacheLoadModifier::LOAD_LDG,
                  "The memory consistency model does not apply to texture "
                  "accesses");
  }
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION SegmentedScanPolicyWrapper<PolicyT> MakeSegmentedScanPolicyWrapper(PolicyT policy)
{
  return SegmentedScanPolicyWrapper<PolicyT>{policy};
}

template <typename InputValueT, typename OutputValueT, typename AccumT, typename OffsetT, typename ScanOpT>
struct policy_hub
{
private:
  using scan_hub = detail::scan::policy_hub<InputValueT, OutputValueT, AccumT, OffsetT, ScanOpT>;

  // reuse policy_hub for scan algorithms. This approach assumes that
  // policy chain here matches the chain used in scan_hub
  template <typename ComputeT, typename AgentScanPolicyT>
  using translate_agent =
    AgentSegmentedScanPolicy<AgentScanPolicyT::BLOCK_THREADS,
                             AgentScanPolicyT::ITEMS_PER_THREAD,
                             ComputeT,
                             AgentScanPolicyT::LOAD_ALGORITHM,
                             AgentScanPolicyT::LOAD_MODIFIER,
                             AgentScanPolicyT::STORE_ALGORITHM,
                             AgentScanPolicyT::SCAN_ALGORITHM>;

public:
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy500::ScanPolicyT>;
  };

  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy520::ScanPolicyT>;
  };

  struct DefaultPolicy
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::DefaultPolicy::ScanPolicyT>;
  };

  struct Policy600
      : DefaultPolicy
      , ChainedPolicy<600, Policy600, Policy520>
  {};

  struct Policy750 : ChainedPolicy<750, Policy750, Policy600>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy750::ScanPolicyT>;
  };

  struct Policy800 : ChainedPolicy<800, Policy800, Policy750>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy800::ScanPolicyT>;
  };

  struct Policy860
      : DefaultPolicy
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy900::ScanPolicyT>;
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    using SegmentedScanPolicyT = translate_agent<AccumT, typename scan_hub::Policy1000::ScanPolicyT>;
  };

  using MaxPolicy = Policy1000;
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
