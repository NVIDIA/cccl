//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <cuda/atomic>
#include <cuda/std/cassert>

#include "common.h"

__host__ __device__ void validate_not_lock_free()
{
  cuda::std::atomic<big_not_lockfree_type> test;
  assert(!test.is_lock_free());
}

void kernel_invoker()
{
  validate_pinned<cuda::std::atomic<big_not_lockfree_type>, basic_testers>();
}

int main(int arg, char** argv)
{
  validate_not_lock_free();

  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
