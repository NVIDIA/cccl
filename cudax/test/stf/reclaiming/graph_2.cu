//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void kernel()
{
  // No-op
}

int main(int argc, char** argv)
{
  int nblocks       = 4;
  size_t block_size = 1024 * 1024;

  if (argc > 1)
  {
    nblocks = atoi(argv[1]);
  }

  if (argc > 2)
  {
    block_size = atoi(argv[2]);
  }

  // At most 1 buffer is allocated at the same time
  setenv("MAX_ALLOC_CNT", "2", 1);

  graph_ctx ctx;
  ::std::vector<logical_data<slice<char>>> handles(nblocks);
  std::vector<char> h_buffer(nblocks * block_size);

  for (int i = 0; i < nblocks; i++)
  {
    handles[i] = ctx.logical_data(make_slice(&h_buffer[i * block_size], block_size));
    handles[i].set_symbol("D_" + std::to_string(i));
  }

  // We only 2 buffers, we are forced to reuse the buffer from D0 for D2
  for (int i = 0; i < 3; i++)
  {
    ctx.task(handles[i % nblocks].read(), handles[(i + 1) % nblocks].rw())
        ->*[&](cudaStream_t s, auto /*unused*/, auto /*unused*/) {
              kernel<<<1, 1, 0, s>>>();
            };
  }

  ctx.submit();

  if (argc > 3)
  {
    std::cout << "Generating DOT output in " << argv[3] << std::endl;
    ctx.print_to_dot(argv[3]);
  }

  ctx.finalize();
}
