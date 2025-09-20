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
 * @brief This test makes sure we can generate a dot file with sections and stackable contexts
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
// TODO (miscco): Make it work for windows
#if !_CCCL_COMPILER(MSVC)
  // Test configuration constants
  constexpr size_t buffer_size    = 64;
  constexpr size_t num_iterations = 3;

  stackable_ctx ctx;

  // Create logical data for the computation pipeline
  auto lA = ctx.logical_data(shape_of<slice<char>>(buffer_size));
  auto lB = ctx.logical_data(shape_of<slice<char>>(buffer_size));
  auto lC = ctx.logical_data(shape_of<slice<char>>(buffer_size));

  // Initialize all logical data in a dedicated DOT section
  auto r_init = ctx.dot_section("init");
  ctx.task(lA.write()).set_symbol("initA")->*[](cudaStream_t, auto) {};
  ctx.task(lB.write()).set_symbol("initB")->*[](cudaStream_t, auto) {};
  ctx.task(lC.write()).set_symbol("initC")->*[](cudaStream_t, auto) {};
  r_init.end();

  // Test nested DOT sections with stackable contexts
  // This creates a hierarchical structure to verify DOT graph generation
  for (size_t j = 0; j < num_iterations; j++)
  {
    auto r0 = ctx.dot_section("lvl0");
    ctx.task(lA.rw()).set_symbol("f1")->*[](cudaStream_t, auto) {};

    ctx.push();
    {
      auto r1 = ctx.dot_section("lvl1");
      ctx.task(lA.read(), lB.rw()).set_symbol("f2")->*[](cudaStream_t, auto, auto) {};
      ctx.task(lA.read(), lC.rw()).set_symbol("f2")->*[](cudaStream_t, auto, auto) {};
      ctx.task(lB.read(), lC.read(), lA.rw()).set_symbol("f3")->*[](cudaStream_t, auto, auto, auto) {};
    }
    ctx.pop();
  }
  ctx.finalize();

#endif // !_CCCL_COMPILER(MSVC)
}
