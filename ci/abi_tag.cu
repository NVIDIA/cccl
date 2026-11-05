// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone compiler probe: deliberately use the attribute without feature guards
// or CCCL headers so unsupported compilers fail instead of silently skipping it.
__attribute__((abi_tag("probe"))) int plain_function(int value)
{
  return value;
}

template <class T>
__attribute__((abi_tag("probe"))) T plain_template(T value)
{
  return value;
}

__host__ __attribute__((abi_tag("probe"))) int host_function(int value)
{
  return value;
}

template <class T>
__host__ __attribute__((abi_tag("probe"))) T host_template(T value)
{
  return value;
}

__device__ __attribute__((abi_tag("probe"))) int device_function(int value)
{
  return value;
}

template <class T>
__device__ __attribute__((abi_tag("probe"))) T device_template(T value)
{
  return value;
}

__host__ __device__ __attribute__((abi_tag("probe"))) int host_device_function(int value)
{
  return value;
}

template <class T>
__host__ __device__ __attribute__((abi_tag("probe"))) T host_device_template(T value)
{
  return value;
}

// Calls instantiate every template, including both halves of host/device templates.
int host_entry(int value)
{
  return plain_function(value) + plain_template(value) + host_function(value) + host_template(value)
       + host_device_function(value) + host_device_template(value);
}

__global__ void device_entry(int* result, int value)
{
  *result = device_function(value) + device_template(value) + host_device_function(value) + host_device_template(value);
}
