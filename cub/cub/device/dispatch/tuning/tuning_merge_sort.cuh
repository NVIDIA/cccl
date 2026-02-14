// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_merge_sort.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::merge_sort
{
template <typename PolicyT, typename = void>
struct MergeSortPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE MergeSortPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct MergeSortPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::MergeSortPolicy::LOAD_MODIFIER)>>
    : StaticPolicyT
{
  _CCCL_HOST_DEVICE MergeSortPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(MergeSort);

#if defined(CUB_ENABLE_POLICY_PTX_JSON)
  _CCCL_DEVICE static constexpr auto EncodedPolicy()
  {
    using namespace ptx_json;
    return object<key<"MergeSortPolicy">() = MergeSort().EncodedPolicy()>();
  }
#endif
};

template <typename PolicyT>
_CCCL_HOST_DEVICE MergeSortPolicyWrapper<PolicyT> MakeMergeSortPolicyWrapper(PolicyT policy)
{
  return MergeSortPolicyWrapper<PolicyT>{policy};
}

template <typename KeyIteratorT>
struct policy_hub
{
  using KeyT = it_value_t<KeyIteratorT>;

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(11),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  // NVBug 3384810
#if defined(_NVHPC_CUDA)
  using Policy520 = Policy500;
#else
  struct Policy520 : ChainedPolicy<520, Policy520, Policy500>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<512,
                           Nominal4BItemsToItems<KeyT>(15),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_LDG,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };
#endif

  struct Policy600 : ChainedPolicy<600, Policy600, Policy520>
  {
    using MergeSortPolicy =
      AgentMergeSortPolicy<256,
                           Nominal4BItemsToItems<KeyT>(17),
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_DEFAULT,
                           BLOCK_STORE_WARP_TRANSPOSE>;
  };

  using MaxPolicy = Policy600;
};
} // namespace detail::merge_sort

CUB_NAMESPACE_END
