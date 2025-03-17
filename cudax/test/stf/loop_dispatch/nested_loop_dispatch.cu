//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cmath>

#include <cuda/experimental/__stf/places/loop_dispatch.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  auto lB = ctx.logical_data<int>(size_t(1024 * 1024));

  ctx.parallel_for(lB.shape(), lB.write())->*[] __device__(size_t i, auto b) {
    b(i) = 42;
  };

  // A fake grid which should work regardless of the underlying machine
  auto grid = exec_place::repeat(exec_place::current_device(), 8);

  // Split the affinity into 4 parts
  loop_dispatch(ctx, grid, place_partition_scope::cuda_device, 0, 4, [&](size_t) {
    // We should have 2 places per subplace
    EXPECT(ctx.current_affinity().size() == 2);

    // This should use ctx.current_affinity() implicitly
    loop_dispatch(ctx, 0, 4, [&](size_t) {
      auto lA = ctx.logical_data<int>(size_t(1024 * 1024));

      ctx.parallel_for(ctx.current_exec_place(), lA.shape(), lA.write())->*[] __device__(size_t i, auto a) {
        a(i) = (int) (10.0 * cuda::std::cos((double) i));
      };

      ctx.parallel_for(ctx.current_exec_place(), lA.shape(), lA.rw(), lB.read())
          ->*[] __device__(size_t i, auto a, auto b) {
                a(i) += b(i);
              };
    });
  });

  ctx.finalize();
}
