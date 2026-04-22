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

#include <cub/thread/thread_load.cuh>
#include <cub/util_device.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
struct find_policy
{
  int block_threads;
  int items_per_thread;
  int vector_load_length;
  CacheLoadModifier load_modifier;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const find_policy& lhs, const find_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.vector_load_length == rhs.vector_load_length && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const find_policy& lhs, const find_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const find_policy& p)
  {
    return os << "find_policy { .block_threads = " << p.block_threads << ", .items_per_thread = " << p.items_per_thread
              << ", .vector_load_length = " << p.vector_load_length << ", .load_modifier = " << p.load_modifier << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept find_policy_selector = policy_selector<T, find_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int input_type_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> find_policy
  {
    // FindPolicy (GTX670: 154.0 @ 48M 4B items) - single policy for all arches
    const auto scaled = scale_mem_bound(128, 16, input_type_size);
    return find_policy{scaled.block_threads, scaled.items_per_thread, 4, LOAD_LDG};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(find_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename InputType>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> find_policy
  {
    return policy_selector{int{sizeof(InputType)}}(arch);
  }
};
} // namespace detail::find

CUB_NAMESPACE_END
