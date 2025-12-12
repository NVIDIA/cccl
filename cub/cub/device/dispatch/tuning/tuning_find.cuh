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

#include <cub/agent/agent_find.cuh>
#include <cub/util_device.cuh>

#include <cuda/std/__iterator/iterator_traits.h>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
template <typename InputIt>
struct policy_hub_t
{
  /// SM30
  struct policy_300_t : ChainedPolicy<300, policy_300_t, policy_300_t>
  {
    static constexpr int threads_per_block  = 128;
    static constexpr int items_per_thread   = 16;
    static constexpr int items_per_vec_load = 4;

    // FindPolicy (GTX670: 154.0 @ 48M 4B items)
    using FindPolicy =
      agent_find_policy_t<threads_per_block,
                          items_per_thread,
                          typename ::cuda::std::iterator_traits<InputIt>::value_type,
                          items_per_vec_load,
                          LOAD_LDG>;
  };

  using MaxPolicy = policy_300_t;
};
} // namespace detail::find

CUB_NAMESPACE_END
