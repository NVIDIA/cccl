/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include "catch2_test_helper.h"
#include "catch2_test_launch_helper.h"

CUB_NAMESPACE_BEGIN

CUB_DETAIL_KERNEL_ATTRIBUTES void write_ptx_version_kernel(int* d_kernel_cuda_arch)
{
  *d_kernel_cuda_arch = CUB_PTX_ARCH;
}

CUB_RUNTIME_FUNCTION static cudaError_t get_cuda_arch_from_kernel(
  void* d_temp_storage, size_t& temp_storage_bytes, int* d_kernel_cuda_arch, int* ptx_version, cudaStream_t stream = 0)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  write_ptx_version_kernel<<<1, 1, 0, stream>>>(d_kernel_cuda_arch);
  return cub::PtxVersion(*ptx_version);
}

CUB_NAMESPACE_END

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::get_cuda_arch_from_kernel, get_cuda_arch_from_kernel);

CUB_TEST("CUB correctly identifies the ptx version the kernel was compiled for", "[util][dispatch]")
{
  constexpr std::size_t single_item = 1;
  c2h::device_vector<int> cuda_arch(single_item);

  // Query the arch the kernel was actually compiled for
  int ptx_version = [&]() -> int {
    int* buffer{};
    cudaMallocHost(&buffer, sizeof(*buffer));
    get_cuda_arch_from_kernel(thrust::raw_pointer_cast(cuda_arch.data()), buffer);
    int result = *buffer;
    cudaFreeHost(buffer);
    return result;
  }();

  int kernel_cuda_arch = cuda_arch[0];

  // Host cub::PtxVersion
  int host_ptx_version{};
  cub::PtxVersion(host_ptx_version);

  // Ensure variable was properly populated
  REQUIRE(0 != kernel_cuda_arch);

  // Ensure that the ptx version corresponds to the arch the kernel was compiled for
  REQUIRE(ptx_version == kernel_cuda_arch);
  REQUIRE(host_ptx_version == kernel_cuda_arch);
}
