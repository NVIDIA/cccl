//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/functional>
#include <cuda/std/cassert>

#include "test_macros.h"

__device__ int scalar_object             = 42;
__device__ const int const_scalar_object = 42;

__device__ int array_object[]             = {42, 1337, -1};
__device__ const int const_array_object[] = {42, 1337, -1};

template <class T>
void test(T& object)
{
  {
    T* address = cuda::std::addressof(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, address);
    assert(status == cudaSuccess);
    assert(attributes.devicePointer == nullptr);
  }

  {
    T* device_address = cuda::get_device_address(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
    assert(status == cudaSuccess);
    assert(attributes.devicePointer == device_address);
  }
}

int main(int argc, char** argv)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (test(scalar_object); test(const_scalar_object); test(array_object); test(const_array_object);),
                    (unused(scalar_object, const_scalar_object, array_object, const_array_object);))

  return 0;
}
