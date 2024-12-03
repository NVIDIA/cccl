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

#include <cuda/std/span>

#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/stream.cuh>

#include "vector.cuh"

namespace cudax = cuda::experimental;
using cudax::in;
using cudax::out;

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(cudax::span<const float> A, cudax::span<const float> B, cudax::span<float> C)
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
  printf("[Vector addition of %d elements]\n", numElements);

  // Allocate the host vectors
  cudax::vector<float> A(numElements); // input
  cudax::vector<float> B(numElements); // input
  cudax::vector<float> C(numElements); // output

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i)
  {
    A[i] = rand() / (float) RAND_MAX;
    B[i] = rand() / (float) RAND_MAX;
  }

  // Define the kernel launch parameters
  constexpr int threadsPerBlock = 256;
  auto dims                     = cudax::distribute<threadsPerBlock>(numElements);

  // Launch the vectorAdd kernel
  printf("CUDA kernel launch with %d blocks of %d threads\n", dims.count(cudax::block, cudax::grid), threadsPerBlock);
  cudax::launch(stream, dims, vectorAdd, in(A), in(B), out(C));

  printf("waiting for the stream to finish\n");
  stream.wait();

  printf("veryfying the results\n");
  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i)
  {
    if (fabs(A[i] + B[i] - C[i]) > 1e-5)
    {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  printf("Done\n");
  return 0;
}
catch (const std::exception& e)
{
  printf("caught an exception: \"%s\"\n", e.what());
}
catch (...)
{
  printf("caught an unknown exception\n");
}
