//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/hierarchy>

static constexpr auto meta_grid_hierarchy =
  cuda::make_hierarchy(cuda::grid_dims(cuda::at_least{1025}, cuda::gpu_thread), cuda::block_dims<256>());
static_assert(cuda::gpu_thread.static_count(cuda::block, meta_grid_hierarchy) == 256);

static constexpr auto meta_cluster_hierarchy = cuda::make_hierarchy(
  cuda::grid_dims<2>(), cuda::cluster_dims(cuda::at_least{4}, cuda::block), cuda::block_dims<128>());
static_assert(cuda::gpu_thread.static_count(cuda::block, meta_cluster_hierarchy) == 128);

static constexpr auto fill_device_hierarchy = cuda::make_hierarchy(cuda::fill_device(), cuda::block_dims<256>());
static_assert(cuda::gpu_thread.static_count(cuda::block, fill_device_hierarchy) == 256);

int main(int, char**)
{
  return 0;
}
