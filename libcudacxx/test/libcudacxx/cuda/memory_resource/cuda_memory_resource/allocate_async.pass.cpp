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

void ensure_device_ptr(void* ptr)
{
  assert(ptr != nullptr);
  cudaPointerAttributes attributes;
  cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  assert(status == cudaSuccess);
  assert(attributes.type == cudaMemoryTypeDevice);
}

void test()
{
#if CUDART_VERSION >= 11020
  cuda::mr::cuda_memory_resource res{};

  ::cudaStream_t stream{};

  { // allocate / deallocate
    auto* ptr = res.allocate_async(42, cuda::mr::default_cuda_malloc_alignment, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate_async(ptr, 42, cuda::mr::default_cuda_malloc_alignment, stream);
  }

  { // allocate / deallocate with alignment
    constexpr size_t desired_alignment = 64;
    auto* ptr                          = res.allocate_async(42, desired_alignment, stream);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    // also check the alignment
    const auto address   = reinterpret_cast<cuda::std::uintptr_t>(ptr);
    const auto alignment = address & (~address + 1ULL);
    assert(alignment >= desired_alignment);
    res.deallocate(ptr, 42, desired_alignment);
  }

#  ifndef TEST_HAS_NO_EXCEPTIONS
  { // allocate with too small alignment
    while (true)
    {
      try
      {
        auto* ptr = res.allocate_async(5, 42, stream);
        unused(ptr);
      }
      catch (const std::bad_alloc&)
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
        auto* ptr = res.allocate_async(5, 1337, stream);
        unused(ptr);
      }
      catch (const std::bad_alloc&)
      {
        break;
      }
      assert(false);
    }
  }
#  endif // TEST_HAS_NO_EXCEPTIONS
#endif // CUDART_VERSION >= 11020
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
