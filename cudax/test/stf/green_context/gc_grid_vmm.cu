//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Test that a grid of green contexts with explicit affine data places
 *        (green context data place extension) and a blocked policy forces
 *        the VMM path for composite allocation.
 *
 * When the grid is made of exec_place::green_ctx(..., true), each place's
 * affine data place is the green context extension. Forming a composite
 * data place from that grid and using the modular API (cdp.allocate(n),
 * cdp.deallocate(ptr, n)) triggers the localized_array (VMM) path.
 */

#include <cuda/experimental/__stf/places/blocked_partition.cuh>
#include <cuda/experimental/__stf/places/exec/green_context.cuh>
#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

#if _CCCL_CTK_AT_LEAST(12, 4)
__global__ void init_kernel(double* ptr, size_t n)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    ptr[i] = static_cast<double>(i);
  }
}

void run_gc_grid_vmm_test()
{
  async_resources_handle handle;
  const int num_sms = 8;
  const int dev_id  = 0;
  auto gc_helper    = handle.get_gc_helper(dev_id, num_sms);

  if (gc_helper->get_count() < 1)
  {
    fprintf(stderr, "No green contexts available, skipping VMM path test.\n");
    return;
  }

  // Grid of green context places with explicit affine data places (VMM path).
  auto where = gc_helper->get_grid(true);

  // Composite data place with blocked partitioner: allocation on this place
  // uses the VMM path (localized_array).
  data_place cdp = data_place::composite(blocked_partition(), where);

  EXPECT(cdp.is_composite());
  EXPECT(!cdp.allocation_is_stream_ordered());

  // 8MB allocation to have enough blocks to test the VMM path.
  const size_t n          = 1024 * 1024;
  const size_t size_bytes = n * sizeof(double);

  // Modular API: allocate directly on the composite place (VMM path)
  void* ptr = cdp.allocate(size_bytes);
  EXPECT(ptr != nullptr);

  double* d_ptr = static_cast<double*>(ptr);
  // Launch on default stream
  int block = 256;
  int grid  = static_cast<int>((n + block - 1) / block);
  init_kernel<<<grid, block>>>(d_ptr, n);
  cuda_safe_call(cudaDeviceSynchronize());

  // Check initialized values: load entire buffer and verify every entry
  ::std::vector<double> h_buf(n);
  cuda_safe_call(cudaMemcpy(h_buf.data(), d_ptr, size_bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(h_buf[i] == static_cast<double>(i));
  }

  // Modular API: deallocate
  cdp.deallocate(ptr, size_bytes);
}
#endif // _CCCL_CTK_AT_LEAST(12, 4)

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Green contexts are not supported by this version of CUDA: skipping test.\n");
  return 0;
#else
  run_gc_grid_vmm_test();
  return 0;
#endif
}
