//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: enable with nvrtc
// UNSUPPORTED: nvrtc

#include <cuda/hierarchy>

struct MyLevel : cuda::hierarchy_level_base<MyLevel>
{};

template <class FromLevel, class ToLevel>
inline constexpr bool trait_v = cuda::__is_natively_reachable_hierarchy_level_v<FromLevel, ToLevel>;

static_assert(!trait_v<cuda::thread_level, cuda::thread_level>);
static_assert(!trait_v<cuda::thread_level, cuda::warp_level>);
static_assert(trait_v<cuda::thread_level, cuda::block_level>);
static_assert(trait_v<cuda::thread_level, cuda::cluster_level>);
static_assert(trait_v<cuda::thread_level, cuda::grid_level>);
static_assert(!trait_v<cuda::thread_level, MyLevel>);

static_assert(!trait_v<cuda::warp_level, cuda::thread_level>);
static_assert(!trait_v<cuda::warp_level, cuda::warp_level>);
static_assert(trait_v<cuda::warp_level, cuda::block_level>);
static_assert(trait_v<cuda::warp_level, cuda::cluster_level>);
static_assert(trait_v<cuda::warp_level, cuda::grid_level>);
static_assert(!trait_v<cuda::warp_level, MyLevel>);

static_assert(!trait_v<cuda::block_level, cuda::thread_level>);
static_assert(!trait_v<cuda::block_level, cuda::warp_level>);
static_assert(!trait_v<cuda::block_level, cuda::block_level>);
static_assert(trait_v<cuda::block_level, cuda::cluster_level>);
static_assert(trait_v<cuda::block_level, cuda::grid_level>);
static_assert(!trait_v<cuda::block_level, MyLevel>);

static_assert(!trait_v<cuda::cluster_level, cuda::thread_level>);
static_assert(!trait_v<cuda::cluster_level, cuda::warp_level>);
static_assert(!trait_v<cuda::cluster_level, cuda::block_level>);
static_assert(!trait_v<cuda::cluster_level, cuda::cluster_level>);
static_assert(trait_v<cuda::cluster_level, cuda::grid_level>);
static_assert(!trait_v<cuda::cluster_level, MyLevel>);

static_assert(!trait_v<cuda::grid_level, cuda::thread_level>);
static_assert(!trait_v<cuda::grid_level, cuda::warp_level>);
static_assert(!trait_v<cuda::grid_level, cuda::block_level>);
static_assert(!trait_v<cuda::grid_level, cuda::cluster_level>);
static_assert(!trait_v<cuda::grid_level, cuda::grid_level>);
static_assert(!trait_v<cuda::grid_level, MyLevel>);

int main(int, char**)
{
  return 0;
}
