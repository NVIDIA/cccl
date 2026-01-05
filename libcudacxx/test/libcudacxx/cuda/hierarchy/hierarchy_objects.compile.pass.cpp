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
#include <cuda/std/type_traits>

static_assert(cuda::std::is_same_v<cuda::thread_level, cuda::std::remove_const_t<decltype(cuda::gpu_thread)>>);
static_assert(cuda::std::is_same_v<cuda::warp_level, cuda::std::remove_const_t<decltype(cuda::warp)>>);
static_assert(cuda::std::is_same_v<cuda::block_level, cuda::std::remove_const_t<decltype(cuda::block)>>);
static_assert(cuda::std::is_same_v<cuda::grid_level, cuda::std::remove_const_t<decltype(cuda::grid)>>);

int main(int, char**)
{
  return 0;
}
