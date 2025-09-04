//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//!
//! \brief Test the behavior of the get_stream() method of the tasks in the different backends

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

void test_stream()
{
  // stream context
  context ctx;

  auto token = ctx.token();
  auto t     = ctx.task(token.write());
  t.start();
  cudaStream_t s = t.get_stream();
  EXPECT(s != nullptr);
  dummy<<<1, 1, 0, s>>>();
  t.end();
  ctx.finalize();
}

void test_graph()
{
  context ctx = graph_ctx();

  auto token = ctx.token();
  auto t     = ctx.task(token.write());
  t.start();
  cudaStream_t s = t.get_stream();
  // We are not capturing so there is no stream associated
  EXPECT(s == nullptr);
  t.end();

  auto t2 = ctx.task(token.rw());
  t2.enable_capture();
  t2.start();
  cudaStream_t s2 = t2.get_stream();
  // We are capturing so the stream used for capture is associated to the task
  EXPECT(s2 != nullptr);
  t2.end();

  ctx.finalize();
}

int main()
{
  test_stream();
  test_graph();
  return 0;
}
