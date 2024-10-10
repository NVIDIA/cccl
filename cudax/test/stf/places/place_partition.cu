//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/places/place_partition.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

void print_partition(async_resources_handle& handle, exec_place place, place_partition_scope scope)
{
  fprintf(stderr, "-----------\n");
  fprintf(
    stderr, "PARTITION %s (scope: %s):\n", place.to_string().c_str(), place_partition_scope_to_string(scope).c_str());
  for (auto sub_place : place_partition(handle, place, scope))
  {
    fprintf(stderr, "[%s] subplace: %s\n", place.to_string().c_str(), sub_place.to_string().c_str());
  }

  fprintf(stderr, "-----------\n");
}

int main()
{
#if CUDA_VERSION < 12040
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  async_resources_handle handle;

  print_partition(handle, exec_place::all_devices(), place_partition_scope::cuda_device);

  print_partition(handle, exec_place::all_devices(), place_partition_scope::cuda_stream);

  print_partition(handle, exec_place::current_device(), place_partition_scope::cuda_stream);

  print_partition(handle, exec_place::current_device(), place_partition_scope::green_context);
  print_partition(handle, exec_place::current_device(), place_partition_scope::green_context);

  print_partition(handle, exec_place::repeat(exec_place::current_device(), 4), place_partition_scope::green_context);

  print_partition(handle, exec_place::current_device(), place_partition_scope::cuda_device);

  print_partition(handle, exec_place::repeat(exec_place::current_device(), 4), place_partition_scope::cuda_stream);
#endif
}
