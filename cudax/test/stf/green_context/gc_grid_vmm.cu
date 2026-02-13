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

  // Build a grid of green context exec places with *explicit* affine data place
  // (use_green_ctx_data_place = true), so each place's affine data place is
  // the green context extension rather than the default device data place.
  ::std::vector<exec_place> places;
  for (size_t i = 0; i < gc_helper->get_count(); i++)
  {
    places.push_back(exec_place::green_ctx(gc_helper->get_view(i), true));
  }

  auto where = make_grid(places);

  // Composite data place with blocked partitioner: allocation on this place
  // uses the VMM path (localized_array).
  data_place cdp = data_place::composite(blocked_partition(), where);

  EXPECT(cdp.is_composite());
  EXPECT(!cdp.allocation_is_stream_ordered());

  const size_t n          = 1024 * 1024; // 1M elements
  const size_t size_bytes = n * sizeof(double);

  // Modular API: allocate directly on the composite place (VMM path)
  void* ptr = cdp.allocate(static_cast<::std::ptrdiff_t>(size_bytes));
  EXPECT(ptr != nullptr);

  // Use the buffer: launch a kernel on the first grid place's stream
  exec_place first_place   = where.get_place(pos4(0));
  decorated_stream dstream = first_place.getStream(handle, false);
  cudaStream_t stream      = dstream.stream;

  double* d_ptr = static_cast<double*>(ptr);
  int block     = 256;
  int grid      = static_cast<int>((n + block - 1) / block);
  init_kernel<<<grid, block, 0, stream>>>(d_ptr, n);

  cuda_safe_call(cudaStreamSynchronize(stream));

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
