//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <template <class> class Wrapper>
__host__ __device__ void test_managed_conversions()
{
  using HostOrDeviceAccessor = Wrapper<cuda::std::default_accessor<float>>;
  using ManagedAccessor      = cuda::managed_accessor<cuda::std::default_accessor<float>>;
  ManagedAccessor host_acc1{HostOrDeviceAccessor{}};
}

template <template <class> class WrapperA, template <class> class WrapperB>
__host__ __device__ void test_host_device_conversions()
{
  using HostOrDeviceAccessor = WrapperA<cuda::std::default_accessor<float>>;
  using DeviceOrHostAccessor = WrapperB<cuda::std::default_accessor<float>>;
  HostOrDeviceAccessor host_acc1{DeviceOrHostAccessor{}};
  DeviceOrHostAccessor host_acc2{HostOrDeviceAccessor{}};
}

int main(int, char**)
{
  test_managed_conversions<cuda::host_accessor>();
  test_managed_conversions<cuda::device_accessor>();
  test_host_device_conversions<cuda::host_accessor, cuda::device_accessor>();
  test_host_device_conversions<cuda::device_accessor, cuda::host_accessor>();
  return 0;
}
