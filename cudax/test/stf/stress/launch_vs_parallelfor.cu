//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/cyclic_shape.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/stopwatch.cuh>
#include <cuda/experimental/stf.cuh>

#include <stdio.h>
#include <time.h>

#define MAX_ITER 200

using namespace cuda::experimental::stf;

int main(int argc, char** argv)
{
  stream_ctx ctx;

  int N0 = 128;
  if (argc > 2)
  {
    N0 = atoi(argv[2]);
  }

  // fprintf(stderr, "Using %d...\n", N0);

  size_t N = size_t(N0) * 1024 * 1024;

#if 0
    auto number_devices = 1;
    auto all_devs = exec_place::repeat<blocked_partition>(exec_place::device(0), number_devices);
#else
  auto all_devs = exec_place::all_devices();
#endif

  data_place cdp;
  // if (argc > 2) {
  //     switch (atoi(argv[2])) {
  //     case 0: cdp = data_place::composite(blocked_partition(), all_devs); break;
  //     case 1: cdp = data_place::managed; break;
  //     case 2: cdp = data_place::device(0); break;
  //     case 3: cdp = data_place::host; break;
  //     default: abort();
  //     }
  // } else {
  cdp = data_place::composite(blocked_partition(), all_devs);
  // }

  fprintf(stderr, "data place: %s\n", cdp.to_string().c_str());

  // data_place cdp = data_place::device(0);
  // data_place cdp = data_place::managed;

  auto data_logical = ctx.logical_data<int>(N);
  // initialize centroids
  ctx.parallel_for(blocked_partition(), all_devs, data_logical.shape(), data_logical.write(cdp))
      ->*[=] _CCCL_DEVICE(size_t ind, auto data) {
            data(ind) = 0;
          };
  int cur_iter = 1;

  const char* const method = argc >= 2 ? argv[1] : "launch-partition";

  stopwatch sw(stopwatch::autostart, ctx.task_fence());

  if (strcmp(method, "launch-partition") == 0)
  {
    for (cur_iter = 1; cur_iter < MAX_ITER; ++cur_iter)
    {
      ctx.launch(all_devs, data_logical.write(cdp)).set_symbol("launch assignment")
          ->*[=] _CCCL_DEVICE(auto&& t, auto&& data) {
                for (auto ind : t.apply_partition(shape(data)))
                {
                  data(ind) = 0;
                }
              };
    }
  }
  else if (strcmp(method, "launch") == 0)
  {
    for (cur_iter = 1; cur_iter < MAX_ITER; ++cur_iter)
    {
      ctx.launch(all_devs, data_logical.rw(cdp))->*[=] _CCCL_DEVICE(auto t, auto data) {};
    }
  }
  else if (strcmp(method, "parallel") == 0)
  {
    for (cur_iter = 1; cur_iter < MAX_ITER; ++cur_iter)
    {
      ctx.parallel_for(blocked_partition(), all_devs, data_logical.shape(), data_logical.rw(cdp))
          ->*[=] _CCCL_DEVICE(size_t ind, auto data) {};
    }
  }
  else if (strcmp(method, "parallel-indexed") == 0)
  {
    for (cur_iter = 1; cur_iter < MAX_ITER; ++cur_iter)
    {
      ctx.parallel_for(blocked_partition(), all_devs, data_logical.shape(), data_logical.rw(cdp))
          .set_symbol("parallel for assignment")
          ->*[=] _CCCL_DEVICE(size_t ind, auto data) {
                data(ind) = 0;
              };
    }
  }
  else
  {
    fprintf(stderr, "Must choose one of launch-partition, parallel-indexed, launch, parallel\n");
    return 1;
  }

  sw.stop(ctx.task_fence());

  ctx.finalize();

  printf("Method: %s, elapsed: %f ms\n", argv[1], sw.elapsed().count());
}
