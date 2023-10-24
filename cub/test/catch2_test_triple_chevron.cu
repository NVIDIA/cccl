/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <thrust/count.h>
#include <cub/detail/triple_chevron_launch.cuh>
#include <cuda/std/tuple>

// Has to go after all cub headers. Otherwise, this test won't catch unused
// variables in cub kernels.
#include "catch2_test_cdp_helper.h"
#include "catch2_test_helper.h"

// %PARAM% TEST_CDP cdp 0:1

template <class T>
__global__ void cub_api_example_x2_0_kernel(const T *d_in, T *d_out, int num_items)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_items)
  {
    d_out[i] = d_in[i] * T{2};
  }
}

DECLARE_CDP_WRAPPER(cub_api_example_t::x2_0, x2_0);

CUB_TEST("Triple Chevron launch kernels from host", "[test][utils]") {
  int n = 42;
  thrust::device_vector<int> in(n, 21);
  thrust::device_vector<int> out(n);
  int *d_in  = thrust::raw_pointer_cast(in.data());
  int *d_out = thrust::raw_pointer_cast(out.data());
  const int block_size = 256;
  const int grid_size = (n * block_size - 1) / block_size;
  auto chev = detail::triple_chevron(grid_size, block_size, 0, 0);
  auto doit_fn = [&]<class K, class ... Args>(K k, Args... args) {
      chev.doit(cub_api_example_x2_0_kernel, ... args);
  };
  cdp_launch(doit_fn, d_in, d_out, n);
}
