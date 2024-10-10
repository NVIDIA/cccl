//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "cudastf/__stf/stream/interfaces/hashtable_linearprobing.h"
#include "cudastf/__stf/stream/stream_ctx.h"
#include "cudastf/__stf/utility/dimensions.h"

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  // This constructor automatically initializes an empty hashtable on the host
  hashtable h;
  auto lh = ctx.logical_data(h);

  ctx.parallel_for(box(16), lh.rw())->*[] CUDASTF_DEVICE(size_t i, auto h) {
    uint32_t key   = 10 * i;
    uint32_t value = 17 + i * 14;
    h.insert(key, value);
  };

  ctx.finalize();

  // Thanks to the write-back mechanism on lh, h has been updated
  for (size_t i = 0; i < 16; i++)
  {
    assert(h.get(i * 10) == 17 + i * 14);
  }
}
