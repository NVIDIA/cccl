//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::synchronous_resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

void test_allocate()
{
  { // allocate_sync(size)
    resource<cuda::mr::host_accessible> input{42};
    cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_sync(0, 0) == ref.allocate_sync(0));

    int expected_after_deallocate = 1337;
    ref.deallocate_sync(static_cast<void*>(&expected_after_deallocate), 0);
    assert(input._val == expected_after_deallocate);
  }

  { // allocate_sync(size, alignment)
    resource<cuda::mr::host_accessible> input{42};
    cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible> ref{input};

    // Ensure that we properly pass on the allocate function
    assert(input.allocate_sync(0, 0) == ref.allocate_sync(0, 0));

    int expected_after_deallocate = 1337;
    ref.deallocate_sync(static_cast<void*>(&expected_after_deallocate), 0, 0);
    assert(input._val == expected_after_deallocate);
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_allocate();))

  return 0;
}
