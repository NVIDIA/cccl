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

#include <cub/util_device.cuh>

#include <cuda/__device/arch_traits.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/cmath>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/algorithm>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail::rotate
{
struct kernel_policy
{
  int threads_per_block;
  int blocks_per_sm;
  int tile_bytes;
  int pipeline_stages;
  int max_regs_per_thread;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const kernel_policy& lhs, const kernel_policy& rhs)
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.blocks_per_sm == rhs.blocks_per_sm
        && lhs.tile_bytes == rhs.tile_bytes && lhs.pipeline_stages == rhs.pipeline_stages
        && lhs.max_regs_per_thread == rhs.max_regs_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const kernel_policy& lhs, const kernel_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const kernel_policy& policy)
  {
    return os
        << "kernel_policy { .threads_per_block = " << policy.threads_per_block
        << ", .blocks_per_sm = " << policy.blocks_per_sm << ", .tile_bytes = " << policy.tile_bytes
        << ", .pipeline_stages = " << policy.pipeline_stages
        << ", .max_regs_per_thread = " << policy.max_regs_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

struct short_algorithm_policy
{
  kernel_policy kernel;
  int tiles_per_grab;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const short_algorithm_policy& lhs, const short_algorithm_policy& rhs)
  {
    return lhs.kernel == rhs.kernel && lhs.tiles_per_grab == rhs.tiles_per_grab;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const short_algorithm_policy& lhs, const short_algorithm_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const short_algorithm_policy& policy)
  {
    return os << "short_algorithm_policy { .kernel = " << policy.kernel
              << ", .tiles_per_grab = " << policy.tiles_per_grab << " }";
  }
#endif // _CCCL_HOSTED()
};

struct long_algorithm_policy
{
  kernel_policy kernel;
  int cooperative_grid_safety_margin;
  ::cuda::std::uint32_t max_direct_dependency_distance;
  ::cuda::std::size_t naive_fallback_min_size;
  double naive_fallback_max_fraction;
  ::cuda::std::size_t naive_fallback_min_distance_bytes;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const long_algorithm_policy& lhs, const long_algorithm_policy& rhs)
  {
    return lhs.kernel == rhs.kernel && lhs.cooperative_grid_safety_margin == rhs.cooperative_grid_safety_margin
        && lhs.max_direct_dependency_distance == rhs.max_direct_dependency_distance
        && lhs.naive_fallback_min_size == rhs.naive_fallback_min_size
        && lhs.naive_fallback_max_fraction == rhs.naive_fallback_max_fraction
        && lhs.naive_fallback_min_distance_bytes == rhs.naive_fallback_min_distance_bytes;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const long_algorithm_policy& lhs, const long_algorithm_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const long_algorithm_policy& policy)
  {
    return os
        << "long_algorithm_policy { .kernel = " << policy.kernel
        << ", .cooperative_grid_safety_margin = " << policy.cooperative_grid_safety_margin
        << ", .max_direct_dependency_distance = " << policy.max_direct_dependency_distance
        << ", .naive_fallback_min_size = " << policy.naive_fallback_min_size
        << ", .naive_fallback_max_fraction = " << policy.naive_fallback_max_fraction
        << ", .naive_fallback_min_distance_bytes = " << policy.naive_fallback_min_distance_bytes << " }";
  }
#endif // _CCCL_HOSTED()
};

struct rotate_policy
{
  short_algorithm_policy short_algorithm;
  long_algorithm_policy long_algorithm;

  [[nodiscard]] _CCCL_API constexpr friend bool operator==(const rotate_policy& lhs, const rotate_policy& rhs)
  {
    return lhs.short_algorithm == rhs.short_algorithm && lhs.long_algorithm == rhs.long_algorithm;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool operator!=(const rotate_policy& lhs, const rotate_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const rotate_policy& policy)
  {
    return os << "rotate_policy { .short_algorithm = " << policy.short_algorithm
              << ", .long_algorithm = " << policy.long_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept rotate_policy_selector = policy_selector<T, rotate_policy>;
#endif // _CCCL_HAS_CONCEPTS()

[[nodiscard]] _CCCL_API constexpr auto make_kernel_policy(
  ::cuda::compute_capability cc, int tile_bytes, int pipeline_stages, int max_regs_per_thread) -> kernel_policy
{
  const auto arch         = ::cuda::arch_traits_for(cc);
  const int blocks_per_sm = ::cuda::std::max(
    static_cast<int>(arch.max_shared_memory_per_multiprocessor) / (tile_bytes * pipeline_stages + 1'000), 1);
  const int threads_per_block = ::cuda::prev_power_of_two(
    ::cuda::std::min(arch.max_threads_per_multiprocessor / blocks_per_sm, arch.max_threads_per_block));
  return {threads_per_block, blocks_per_sm, tile_bytes, pipeline_stages, max_regs_per_thread};
}

struct policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> rotate_policy
  {
    constexpr int short_tile_bytes = 18 * 1024;
    constexpr int long_tile_bytes  = 32 * 1024;
    static_assert(short_tile_bytes < long_tile_bytes,
                  "The short rotate tile must be smaller than the long rotate tile");

    return rotate_policy{
      short_algorithm_policy{
        /* .kernel = */ make_kernel_policy(
          cc, /* tile_bytes = */ short_tile_bytes, /* pipeline_stages = */ 2, /* max_regs_per_thread = */ 4),
        /* .tiles_per_grab = */ 6,
      },
      long_algorithm_policy{
        /* .kernel = */ make_kernel_policy(
          cc, /* tile_bytes = */ long_tile_bytes, /* pipeline_stages = */ 2, /* max_regs_per_thread = */ 4),
        /* .cooperative_grid_safety_margin = */ 50,
        /* .max_direct_dependency_distance = */ 256,
        /* .naive_fallback_min_size = */ 1'000'000'000,
        /* .naive_fallback_max_fraction = */ 0.3,
        /* .naive_fallback_min_distance_bytes = */ 500'000'000,
      }};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(rotate_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::rotate
CUB_NAMESPACE_END
