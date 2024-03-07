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
// UNSUPPORTED: nvrtc

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

void ensure_device_ptr(void* ptr) {
    assert(ptr != nullptr);
    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes (&attributes, ptr);
    assert(status == cudaSuccess);
    assert(attributes.type == cudaMemoryTypeDevice );
}

void test() {
  cuda::mr::cuda_memory_resource res{};

  { // allocate / deallocate
    auto* ptr = res.allocate(42);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42);
  }

  { // allocate / deallocate with alignment
    auto* ptr = res.allocate(42, 4);
    static_assert(cuda::std::is_same<decltype(ptr), void*>::value, "");
    ensure_device_ptr(ptr);

    res.deallocate(ptr, 42, 4);
  }
}

int main(int, char**) {
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
