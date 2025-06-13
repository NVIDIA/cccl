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
  auto epoch = ctx.epoch();

  // Begin a top-level section named "foo"
  auto s_foo = ctx.dot_section("foo");
  for (size_t i = 0; i < 2; i++)
  {
    // Section named "bar" using RAII
    auto s_bar = ctx.dot_section("bar");
    ctx.task(epoch).set_symbol("t1")->*[](cudaStream_t) {};
    for (size_t j = 0; j < 2; j++)
    {
      // Section named "baz" using RAII
      auto s_bar = ctx.dot_section("baz");
      ctx.task(epoch).set_symbol("t2")->*[](cudaStream_t) {};
      epoch++;
      ctx.task(epoch).set_symbol("t3")->*[](cudaStream_t) {};
      epoch++;
      // Implicit end of section "baz"
    }
    epoch++;
    // Implicit end of section "bar"
  }
  s_foo.end(); // Explicit end of section "foo"
  ctx.finalize();
#endif // !_CCCL_COMPILER(MSVC)
}
