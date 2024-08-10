/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>

#include <cstdio>
#include <vector>

#include "async_copy.cuh"
#include "async_memory_resource.cuh"
#include "async_uninitialized_buffer.cuh"

namespace cudax = cuda::experimental;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(cuda::std::span<const float> A, cuda::std::span<const float> B, cuda::std::span<float> C)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < A.size())
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void)
try
{
  // A CUDA stream on which to execute the vector addition kernel
  cudax::stream stream(cudax::devices[0]);

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  std::printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host vectors
  std::vector<float> h_A(numElements); // input
  std::vector<float> h_B(numElements); // input
  std::vector<float> h_C(numElements); // output

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i)
  {
    h_A[i] = rand() / (float) RAND_MAX;
    h_B[i] = rand() / (float) RAND_MAX;
  }

  // This is a resource for allocating device memory
  cudax::cuda_async_memory_resource mr(stream.device());

  // Asynchronously allocate the device buffers on the stream
  cudax::async_uninitialized_buffer d_A(float{}, mr, numElements, stream);
  cudax::async_uninitialized_buffer d_B(float{}, mr, numElements, stream);
  cudax::async_uninitialized_buffer d_C(float{}, mr, numElements, stream);

  // Asynchronously copy the host input data to the device
  cudax::async_copy(h_A, d_A, stream);
  cudax::async_copy(h_B, d_B, stream);

  // Define the kernel launch parameters
  constexpr int threadsPerBlock = 256;
  auto dims                     = cudax::distribute<threadsPerBlock>(numElements);

  // Launch the vectorAdd kernel
  std::printf(
    "CUDA kernel launch with %d blocks of %d threads\n", dims.count(cudax::block, cudax::grid), threadsPerBlock);
  cudax::launch(stream, dims, vectorAdd, d_A, d_B, d_C);

  // Copy the result from device to host
  cudax::async_copy(d_C, h_C, stream);

  std::printf("waiting for the stream to finish\n");
  stream.wait();

  std::printf("veryfying the results\n");
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
      std::fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  std::printf("Test PASSED\n");

  std::printf("Done\n");
  return 0;
}
catch (const std::exception& e)
{
  std::printf("caught an exception: \"%s\"\n", e.what());
}
catch (...)
{
  std::printf("caught an unknown exception\n");
}
