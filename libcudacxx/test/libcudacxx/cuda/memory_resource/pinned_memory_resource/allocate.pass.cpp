//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include "test_macros.h"

void ensure_pinned_host_ptr(void* ptr)
{
  assert(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  assert(status == cudaSuccess);
  assert((attributes.type == cudaMemoryTypeHost) && (attributes.devicePointer != nullptr));
}

void test(const unsigned int flag)
{
  cuda::mr::pinned_memory_resource res{flag};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_host_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_pinned_host_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }

#ifndef TEST_HAS_NO_EXCEPTIONS
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 42);
        unused(ptr);
      }
      catch (const std::invalid_argument&)
      {
        break;
      }
      assert(false);
    }
  }

  { // allocate with non matching alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate(5, 1337);
        unused(ptr);
      }
      catch (const std::invalid_argument&)
      {
        break;
      }
      assert(false);
    }
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

void test()
{
  test(cudaHostAllocDefault);
  test(cudaHostAllocPortable);
  test(cudaHostAllocMapped);
  test(cudaHostAllocWriteCombined);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
