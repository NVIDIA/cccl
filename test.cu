//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/cassert>
#include <cuda/stream>

#include <cuda/experimental/hierarchy.cuh>

namespace cudax = cuda::experimental;

template <class... TGArgs>
__device__ int reduce(cudax::thread_group<cuda::block_level, TGArgs...> threads, int value)
{
  return ::__reduce_add_sync(0xffff'ffff, value);
}

struct kernel
{
  template <class Config>
  __device__ void operator()(const Config& config) const noexcept
  {
    // 1. Create thread group within the whole block.
    cudax::thread_group tg1{cuda::block, config};

    // 2. Create thread group within warp that splits to odd and even ranks.
    cudax::thread_group tg2{cuda::warp, config, (cuda::gpu_thread.rank(cuda::warp) % 2) == 0};
    assert(tg2.count(cuda::gpu_thread) == 16);
    if (tg2.group_rank() == 0)
    {
      printf("%d: I am even\n", cuda::gpu_thread.rank_as<int>(cuda::warp));
    }
    else
    {
      printf("%d: I am odd\n", cuda::gpu_thread.rank_as<int>(cuda::warp));
    }

    int init_value            = (tg2.rank(cuda::gpu_thread) == 0) ? 1 : 2;
    const auto exp_init_value = (cuda::gpu_thread.rank(cuda::warp) < 2) ? 1 : 2;
    assert(init_value == exp_init_value);

    // 3. Same as 2., but use lambda to select the group based on the thread rank in warp
    cudax::thread_group tg3{cuda::warp, config, [](unsigned tid) {
                              return tid < 10;
                            }};

    // 4. Creates generic thread group, can result in up to 32 separate groups.
    cudax::thread_group tg4{cuda::warp, config, cuda::gpu_thread.rank_as<unsigned>(cuda::warp)};

    // 5. Splits the threads into groups of 4.
    cudax::thread_group tg5{cuda::warp, config, cudax::group_by<4>{}};

    // A call to cub?
    const auto result = reduce(tg1, cuda::gpu_thread.rank(cuda::warp));
    tg1.sync();

    if (cuda::gpu_thread.rank(cuda::block) == 0)
    {
      printf("Result: %d\n", result);
    }
  }
};

int main()
{
  cuda::stream stream{cuda::devices[0]};

  cuda::launch(stream, cuda::distribute<32>(32), kernel{});

  stream.sync();
}
