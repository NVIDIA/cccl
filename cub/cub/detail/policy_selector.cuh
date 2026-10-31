// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/remove_cvref.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename PolicyTransformer, typename DefaultPolicySelector>
struct policy_selector_from_transformer
{
  _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const
  {
    return PolicyTransformer{}(cc, DefaultPolicySelector{}(cc));
  }
};

template <typename PolicySelectorOrTransformer, typename DefaultPolicySelector>
using to_policy_selector_t =
  ::cuda::std::conditional_t<::cuda::std::is_invocable_v<PolicySelectorOrTransformer, ::cuda::compute_capability>,
                             PolicySelectorOrTransformer,
                             policy_selector_from_transformer<PolicySelectorOrTransformer, DefaultPolicySelector>>;

// Queries the tuning environment for a policy selector or policy transformer registered for PolicyT. If there is none,
// DefaultPolicySelector is used. The result is always a policy selector.
template <typename TuningEnvT, typename PolicyT, typename DefaultPolicySelector>
using policy_selector_from_tuning_env_t = to_policy_selector_t<
  ::cuda::std::remove_cvref_t<::cuda::std::execution::__query_result_or_t<TuningEnvT, PolicyT, DefaultPolicySelector>>,
  DefaultPolicySelector>;
} // namespace detail

CUB_NAMESPACE_END
