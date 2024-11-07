//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

/**
 * @brief Ensure the buddy allocation is working properly on the different backends
 */

using namespace cuda::experimental::stf;

template <typename ctx_t>
void test_buddy()
{
  ctx_t ctx;
  ctx.set_allocator(block_allocator<buddy_allocator>(ctx));

  std::vector<logical_data<slice<char>>> data;
  for (size_t i = 0; i < 10; i++)
  {
    size_t s = (1 + i % 8) * 1024ULL * 1024ULL;
    auto l   = ctx.logical_data(shape_of<slice<char>>(s));
    data.push_back(l);

    ctx.task(l.write())->*[](cudaStream_t, auto) {};
  }

  ctx.finalize();
}

int main(int, char**)
{
  test_buddy<stream_ctx>();
  test_buddy<graph_ctx>();
  test_buddy<context>();
}
