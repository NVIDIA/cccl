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

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>
namespace cudax = cuda::experimental;

namespace cuda::experimental
{
using thrust::device_vector;
using thrust::host_vector;
} // namespace cuda::experimental

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float* A, const float* B, float* C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements)
  {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void)
{
  // A CUDA stream on which to execute the vector addition kernel
  cudax::stream stream;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host input vector A
  cudax::host_vector<float> h_A(numElements);

  // Allocate the host input vector B
  cudax::host_vector<float> h_B(numElements);

  // Allocate the host output vector C
  cudax::host_vector<float> h_C(numElements);

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i)
  {
    h_A[i] = rand() / (float) RAND_MAX;
    h_B[i] = rand() / (float) RAND_MAX;
  }

  // Allocate the device input vector A
  cudax::device_vector<float> d_A(numElements);

  // Allocate the device input vector B
  cudax::device_vector<float> d_B(numElements);

  // Allocate the device output vector C
  cudax::device_vector<float> d_C(numElements);

  // Copy the host input vectors A and B in host memory to the device input
  // vectors in
  // device memory
  printf("Copy input data from the host memory to the CUDA device\n");
  d_A = h_A;
  d_B = h_B;

  // Launch the Vector Add CUDA Kernel
  constexpr int threadsPerBlock = 256;
  int blocksPerGrid             = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  auto dims = cudax::make_hierarchy(cudax::grid_dims(blocksPerGrid), cudax::block_dims<threadsPerBlock>());
  cudax::launch(stream, dims, vectorAdd, d_A.data().get(), d_B.data().get(), d_C.data().get(), numElements);

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the CUDA device to the host memory\n");
  h_C = d_C;

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  printf("Done\n");
  return 0;
}
