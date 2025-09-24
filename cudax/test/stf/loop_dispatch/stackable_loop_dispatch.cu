//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/loop_dispatch.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  // Loop count
  int n = 1024;

  auto lB = ctx.logical_data<int>(size_t(1024 * 1024)).set_symbol("B");
  ctx.parallel_for(lB.shape(), lB.write())->*[] __device__(size_t i, auto b) {
    b(i) = 42;
  };

  lB.set_read_only(true);

  for (size_t iter = 0; iter < 4; iter++)
  {
    loop_dispatch(ctx, exec_place::all_devices(), place_partition_scope::green_context, 0, n, [&](size_t iter) {
      auto lA = ctx.logical_data<int>(size_t(1024 * 1024)).set_symbol(::std::string("A") + ::std::to_string(iter));

      ctx.parallel_for(ctx.current_exec_place(), lA.shape(), lA.write()).set_symbol("pfor1" + ::std::to_string(iter))
          ->*[] __device__(size_t i, auto a) {
                a(i) = (int) (10.0 * cos((double) i));
              };

      ctx.parallel_for(ctx.current_exec_place(), lA.shape(), lA.rw(), lB.read())
          .set_symbol("pfor2" + ::std::to_string(iter))
          ->*[] __device__(size_t i, auto a, auto b) {
                a(i) += b(i);
              };
    });
  }

  ctx.finalize();
}
