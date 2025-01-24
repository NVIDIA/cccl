/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce
{
template <typename PolicyT, typename = void>
struct ReducePolicyWrapper : PolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct ReducePolicyWrapper<StaticPolicyT,
                           _CUDA_VSTD::void_t<typename StaticPolicyT::ReducePolicy,
                                              typename StaticPolicyT::SingleTilePolicy,
                                              typename StaticPolicyT::SegmentedReducePolicy>> : StaticPolicyT
{
  CUB_RUNTIME_FUNCTION ReducePolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  CUB_DEFINE_SUB_POLICY_GETTER(Reduce)
  CUB_DEFINE_SUB_POLICY_GETTER(SingleTile)
  CUB_DEFINE_SUB_POLICY_GETTER(SegmentedReduce)
};

template <typename PolicyT>
CUB_RUNTIME_FUNCTION ReducePolicyWrapper<PolicyT> MakeReducePolicyWrapper(PolicyT policy)
{
  return ReducePolicyWrapper<PolicyT>{policy};
}

template <typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_hub
{
  struct Policy300 : ChainedPolicy<300, Policy300, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 2;

    // ReducePolicy (GTX670: 154.0 @ 48M 4B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_DEFAULT>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  struct Policy350 : ChainedPolicy<350, Policy350, Policy300>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 20;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (GTX Titan: 255.1 GB/s @ 48M 4B items; 228.7 GB/s @ 192M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  struct Policy600 : ChainedPolicy<600, Policy600, Policy350>
  {
    static constexpr int threads_per_block  = 256;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // ReducePolicy (P100: 591 GB/s @ 64M 4B items; 583 GB/s @ 256M 1B items)
    using ReducePolicy =
      AgentReducePolicy<threads_per_block,
                        items_per_thread,
                        AccumT,
                        items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS,
                        LOAD_LDG>;

    using SingleTilePolicy      = ReducePolicy;
    using SegmentedReducePolicy = ReducePolicy;
  };

  using MaxPolicy = Policy600;
};
} // namespace reduce
} // namespace detail

/// @tparam AccumT
///   Accumulator data type
///
/// OffsetT
///   Signed integer type for global offsets
///
/// ReductionOpT
///   Binary reduction functor type having member
///   `auto operator()(const T &a, const U &b)`
template <typename AccumT, typename OffsetT, typename ReductionOpT>
using DeviceReducePolicy CCCL_DEPRECATED_BECAUSE(
  "This class is considered an implementation detail and it will be "
  "removed.") = detail::reduce::policy_hub<AccumT, OffsetT, ReductionOpT>;

template <typename PolicyT, typename Enable = void>
using ReducePolicyWrapper CCCL_DEPRECATED_BECAUSE("This class is considered an implementation detail and it will be "
                                                  "removed.") = detail::reduce::ReducePolicyWrapper<PolicyT, Enable>;

template <typename PolicyT>
CCCL_DEPRECATED_BECAUSE("This function is considered an implementation detail and it will "
                        "be removed.")
CUB_RUNTIME_FUNCTION detail::reduce::ReducePolicyWrapper<PolicyT> MakeReducePolicyWrapper(PolicyT policy)
{
  return detail::reduce::ReducePolicyWrapper<PolicyT>{policy};
}

CUB_NAMESPACE_END
