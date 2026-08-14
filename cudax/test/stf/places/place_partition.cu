//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__stf/internal/stf_places_partition_into_stf.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

void print_partition(async_resources_handle& handle, exec_place place, place_partition_scope scope)
{
  fprintf(stderr, "-----------\n");
  fprintf(stderr, "PARTITION %s (scope: %s):\n", place.to_string().c_str(), place_partition_scope_to_string(scope));
  for (auto sub_place : place_partition(place, handle, scope))
  {
    fprintf(stderr, "[%s] subplace: %s\n", place.to_string().c_str(), sub_place.to_string().c_str());
  }

  fprintf(stderr, "-----------\n");
}

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping handle-based tests.\n");
#else // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 4) vvv
  async_resources_handle handle;

  print_partition(handle, exec_place::all_devices(), place_partition_scope::cuda_device);

  print_partition(handle, exec_place::all_devices(), place_partition_scope::cuda_stream);

  print_partition(handle, exec_place::current_device(), place_partition_scope::cuda_stream);

  print_partition(handle, exec_place::current_device(), place_partition_scope::green_context);
  print_partition(handle, exec_place::current_device(), place_partition_scope::green_context);

  print_partition(handle, exec_place::repeat(exec_place::current_device(), 4), place_partition_scope::green_context);

  print_partition(handle, exec_place::current_device(), place_partition_scope::cuda_device);

  print_partition(handle, exec_place::repeat(exec_place::current_device(), 4), place_partition_scope::cuda_stream);

  // Locality-domain scope: works with or without a handle, on any device
  // (unsupported devices degrade to one whole-device domain, so the count is
  // never zero).
  print_partition(handle, exec_place::current_device(), place_partition_scope::locality_domain);
  print_partition(handle, exec_place::all_devices(), place_partition_scope::locality_domain);
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^

  // No-handle locality-domain tests: this scope works on every toolkit
  // (devices without locality-domain support degrade to one whole-device
  // domain), so these run unguarded.
  const int ndevs = cuda_try<cudaGetDeviceCount>();
  size_t expected = 0;
  for (int d = 0; d < ndevs; d++)
  {
    expected += locality_domain_count(d);
  }

  // Machine-wide grid at locality-domain granularity (no handle needed)
  place_partition machine_domains(exec_place::all_devices(), place_partition_scope::locality_domain);
  EXPECT(machine_domains.size() == expected,
         "all_devices at locality_domain scope: got ",
         machine_domains.size(),
         " subplaces, expected ",
         expected);

  // Single-device partition must match the make_locality_domain_grid sugar
  place_partition dev_domains(exec_place::current_device(), place_partition_scope::locality_domain);
  exec_place sugar = make_locality_domain_grid(0);
  EXPECT(dev_domains.size() == sugar.size(),
         "current_device at locality_domain scope: got ",
         dev_domains.size(),
         " subplaces, expected ",
         sugar.size());
  for (size_t i = 0; i < dev_domains.size(); i++)
  {
    EXPECT((dev_domains.get(i) == sugar.get_place(i)),
           "subplace ",
           i,
           " differs between place_partition and make_locality_domain_grid");
  }
}
