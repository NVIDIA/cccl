//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Experiment with local context nesting
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  int array[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    array[i] = 1 + i * i;
  }

  auto lA = ctx.logical_data(array).set_symbol("A");

  // repeat : {tmp = a; tmp*=2; a+=tmp}
  for (size_t iter = 0; iter < 10; iter++)
  {
    stackable_ctx::graph_scope_guard graph{ctx}; // RAII: automatic push/pop (lock_guard style)

    auto tmp = ctx.logical_data(lA.shape()).set_symbol("tmp");

    ctx.parallel_for(tmp.shape(), tmp.write(), lA.read())->*[] __device__(size_t i, auto tmp, auto a) {
      tmp(i) = a(i);
    };

    ctx.parallel_for(tmp.shape(), tmp.rw())->*[] __device__(size_t i, auto tmp) {
      tmp(i) *= 2;
    };

    ctx.parallel_for(lA.shape(), tmp.read(), lA.rw())->*[] __device__(size_t i, auto tmp, auto a) {
      a(i) += tmp(i);
    };

    // ctx.pop() is called automatically when 'graph' goes out of scope
  }

  ctx.finalize();

  // Verify the array has been updated correctly by the write-back mechanism
  // Each iteration transforms each element: a_new = a_old + 2 * a_old = 3 * a_old
  // Starting from array[i] = 1 + i*i, after 10 iterations:
  // array[i] = 3^10 * (1 + i*i)
  constexpr int pow3_10 = 59049; // 3^10

  for (size_t i = 0; i < 1024; i++)
  {
    int expected = pow3_10 * (1 + static_cast<int>(i * i));
    EXPECT(array[i] == expected);
  }
}
