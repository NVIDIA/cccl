//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

__device__ void device_test()
{
  __shared__ int smem[4];
  assert(cuda::__is_valid_address_range(smem, sizeof(int) * 4));
  assert(!cuda::__is_valid_address_range(smem, 0));
  assert(!cuda::__is_valid_address_range(smem, 64'000'000)); // larger than total smem size
  int var = 0;
  assert(cuda::device::__is_smem_valid_address_range(smem, sizeof(smem)));
  assert(!cuda::device::__is_smem_valid_address_range(smem, 64'000'000));
  assert(!cuda::device::__is_smem_valid_address_range(&var, sizeof(var)));
  assert(!cuda::device::__is_smem_valid_address_range(&var, 64'000'000)); // larger than total smem size
  assert(!cuda::device::__is_smem_valid_address_range(&var, cuda::std::numeric_limits<size_t>::max()));
}

__host__ __device__ void host_device_test()
{
  int var = 0;
  assert(cuda::__is_valid_address_range(&var, sizeof(var)));
  NV_IF_TARGET(NV_PROVIDES_SM_90,
               (assert(!cuda::__is_valid_address_range(nullptr, 1));),
               (assert(cuda::__is_valid_address_range(nullptr, 1));))
  auto ptr2 = reinterpret_cast<int*>(cuda::std::numeric_limits<uintptr_t>::max());
  assert(!cuda::__is_valid_address_range(ptr2, 4));
}

__host__ __device__ bool test()
{
  NV_IF_TARGET(NV_IS_DEVICE, (device_test();))
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
