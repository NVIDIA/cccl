//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/interfaces/hashtable_linearprobing.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>
#include <cuda/experimental/__stf/utility/dimensions.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  // Create a logical data from a shape of hashtable
  auto lh = ctx.logical_data(shape_of<hashtable>());

  // A write() access is needed because we initialized lh from a shape, so there is no reference copy
  ctx.parallel_for(box(16), lh.write())->*[] _CCCL_DEVICE(size_t i, auto h) {
    uint32_t key   = 10 * i;
    uint32_t value = 17 + i * 14;
    h.insert(key, value);
  };

  ctx.host_launch(lh.rw())->*[](auto h) {
    for (int i = 0; i < 16; i++)
    {
      EXPECT(h.get(i * 10) == 17 + i * 14);
    }
  };

  ctx.finalize();
}
