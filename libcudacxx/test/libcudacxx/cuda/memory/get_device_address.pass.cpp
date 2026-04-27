//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

#include <cuda/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int scalar_object             = 42;
TEST_GLOBAL_VARIABLE const int const_scalar_object = 42;

TEST_GLOBAL_VARIABLE int array_object[]             = {42, 1337, -1};
TEST_GLOBAL_VARIABLE const int const_array_object[] = {42, 1337, -1};

#if !TEST_COMPILER(NVRTC)
template <class T>
void test_host(T& object)
{
  {
    T* host_address = cuda::std::addressof(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, host_address);
    assert(status == cudaSuccess);

    if (attributes.devicePointer)
    {
      assert(attributes.devicePointer == host_address);
    }
  }

  {
    T* device_address = cuda::get_device_address(object);

    cudaPointerAttributes attributes;
    cudaError_t status = cudaPointerGetAttributes(&attributes, device_address);
    assert(status == cudaSuccess);
    assert(attributes.devicePointer == device_address);
  }
}
#endif // !TEST_COMPILER(NVRTC)

template <class T>
TEST_FUNC void test(T& object)
{
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE, (assert(cuda::std::addressof(object) == cuda::get_device_address(object));), (test_host(object);))
}

int main(int argc, char** argv)
{
  test(scalar_object);
  test(const_scalar_object);
  test(array_object);
  test(const_array_object);

  return 0;
}
