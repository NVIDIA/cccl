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
 * @brief Ensure we can use the add_deps mechanism on tasks in stackable contexts.
 *        This test verifies that dependencies can be added one by one to tasks
 *        using add_deps() in stackable contexts, with proper automatic data
 *        pushing and validation.
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// a = b + 1;
template <typename T>
__global__ void add(slice<T> a, slice<T> b)
{
  a(0) += b(0);
}

int main()
{
  stackable_ctx ctx;

  int var1 = 42;
  int var2 = 64;

  auto lvar1 = ctx.logical_data(make_slice(&var1, 1));
  auto lvar2 = ctx.logical_data(make_slice(&var2, 1));

  ctx.push();

  auto t = ctx.task();
  t.add_deps(lvar1.rw());
  t.add_deps(lvar2.read());
  t->*[&t](cudaStream_t stream) {
    auto d1 = t.template get<slice<int>>(0);
    auto d2 = t.template get<slice<int>>(1);
    add<<<1, 1, 0, stream>>>(d1, d2);
  };

  ctx.pop();

  ctx.host_launch(lvar1.read())->*[]([[maybe_unused]] auto d1) {
    _CCCL_ASSERT(d1(0) == (42 + 64), "invalid result");
  };

  ctx.finalize();
}
