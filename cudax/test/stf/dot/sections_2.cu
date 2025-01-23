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
 * @brief This test makes sure we can generate a dot file with sections
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
// TODO (miscco): Make it work for windows
#if !_CCCL_COMPILER(MSVC)
  context ctx;
  auto lA = ctx.logical_token().set_symbol("A");
  auto lB = ctx.logical_token().set_symbol("B");
  auto lC = ctx.logical_token().set_symbol("C");
  ctx.dot_push_section("foo");
  for (size_t i = 0; i < 2; i++)
  {
    auto guard = ctx.dot_section("bar");
    ctx.task(lA.read(), lB.rw()).set_symbol("t1")->*[](cudaStream_t, auto, auto) {};
    for (size_t j = 0; j < 2; j++)
    {
      auto inner_guard = ctx.dot_section("baz");
      ctx.task(lA.read(), lC.rw()).set_symbol("t2")->*[](cudaStream_t, auto, auto) {};
      ctx.task(lB.read(), lC.read(), lA.rw()).set_symbol("t3")->*[](cudaStream_t, auto, auto, auto) {};
    }
  }
  ctx.dot_pop_section();
  ctx.finalize();
#endif // !_CCCL_COMPILER(MSVC)
}
