//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define _CUDAX_DISABLE_CG_INTEROP

#include <cuda/experimental/group.cuh>

// Check both toolkit headers before any intentional include of them.
#if defined(_COOPERATIVE_GROUPS_H_) || defined(_CG_INFO_H_)
#  error "group.cuh included cooperative groups despite the interop opt-out"
#endif

#if __has_include(<cooperative_groups.h>)
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>

#  include <cooperative_groups.h>

namespace
{
using namespace cuda::experimental;

template <class From, class Hierarchy>
__device__ auto can_make_cg_equivalent_group(int)
  -> decltype(make_cg_equivalent_group(cuda::std::declval<const From&>(), cuda::std::declval<const Hierarchy&>()),
              cuda::std::true_type{});

template <class From, class Hierarchy>
__device__ cuda::std::false_type can_make_cg_equivalent_group(...);

template <template <class> class Group, class From>
__device__ void check_disabled_interop()
{
  using hierarchy = decltype(implicit_hierarchy());
  static_assert(cuda::std::is_constructible_v<Group<hierarchy>, const hierarchy&>);
  static_assert(!decltype(can_make_cg_equivalent_group<From, hierarchy>(0))::value);
}

[[maybe_unused]] __device__ void test_disabled_interop()
{
  check_disabled_interop<this_thread, cooperative_groups::thread_block_tile<1, void>>();
  check_disabled_interop<this_warp, cooperative_groups::thread_block_tile<32, cooperative_groups::thread_block>>();
  check_disabled_interop<this_block, cooperative_groups::thread_block>();
#  if defined(_CG_HAS_CLUSTER_GROUP)
  check_disabled_interop<this_cluster, cooperative_groups::cluster_group>();
#  endif
  check_disabled_interop<this_grid, cooperative_groups::grid_group>();
}
} // namespace
#endif // __has_include(<cooperative_groups.h>)

#include <c2h/catch2_test_helper.h>

C2H_TEST("group.cuh cooperative-groups interop can be disabled", "[group]") {}
