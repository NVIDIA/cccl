/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <cuda/experimental/device.cuh>

#include <stdio.h>
#include <stdlib.h>

__global__ void SimpleKernel(float* src, float* dst)
{
  // Just a dummy kernel, doing enough for us to verify that everything
  // worked
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  dst[idx]      = src[idx] * 2.0f;
}

int main(int argc, char** argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  printf("CUDA-capable device count: %llu\n", cudax::devices.size());

  if (cudax::devices.size())
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for "
           "%s.\n",
           argv[0]);
    printf("Waiving test.\n");
    exit(2);
  }

  // Query device properties
  cudaDeviceProp prop[64];
  int gpuid[2]; // we want to find the first two GPU's that can support P2P

  for (int i = 0; i < gpu_n; i++)
  {
    checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));
  }
  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  int can_access_peer;
  int p2pCapableGPUs[2]; // We take only 1 pair of P2P capable GPUs
  p2pCapableGPUs[0] = p2pCapableGPUs[1] = -1;

  // Show all the combinations of supported P2P GPUs
  for (int i = 0; i < gpu_n; i++)
  {
    for (int j = 0; j < gpu_n; j++)
    {
      if (i == j)
      {
        continue;
      }
      checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, i, j));
      printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
             prop[i].name,
             i,
             prop[j].name,
             j,
             can_access_peer ? "Yes" : "No");
      if (can_access_peer && p2pCapableGPUs[0] == -1)
      {
        p2pCapableGPUs[0] = i;
        p2pCapableGPUs[1] = j;
      }
    }
  }

  if (p2pCapableGPUs[0] == -1 || p2pCapableGPUs[1] == -1)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for "
           "%s.\n",
           argv[0]);
    printf("Peer to Peer access is not available amongst GPUs in the system, "
           "waiving test.\n");

    exit(EXIT_WAIVED);
  }

  // Use first pair of p2p capable GPUs detected.
  gpuid[0] = p2pCapableGPUs[0];
  gpuid[1] = p2pCapableGPUs[1];

  // Enable peer access
  printf("Enabling peer access between GPU%d and GPU%d...\n", gpuid[0], gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[1], 0));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaDeviceEnablePeerAccess(gpuid[0], 0));

  // Allocate buffers
  const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n", int(buf_size / 1024 / 1024), gpuid[0], gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  float* g0;
  checkCudaErrors(cudaMalloc(&g0, buf_size));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  float* g1;
  checkCudaErrors(cudaMalloc(&g1, buf_size));
  float* h0;
  checkCudaErrors(cudaMallocHost(&h0, buf_size)); // Automatically portable with UVA

  // Create CUDA event handles
  printf("Creating event handles...\n");
  cudaEvent_t start_event, stop_event;
  float time_memcpy;
  int eventflags = cudaEventBlockingSync;
  checkCudaErrors(cudaEventCreateWithFlags(&start_event, eventflags));
  checkCudaErrors(cudaEventCreateWithFlags(&stop_event, eventflags));

  // P2P memcopy() benchmark
  checkCudaErrors(cudaEventRecord(start_event, 0));

  for (int i = 0; i < 100; i++)
  {
    // With UVA we don't need to specify source and target devices, the
    // runtime figures this out by itself from the pointers
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      checkCudaErrors(cudaMemcpy(g1, g0, buf_size, cudaMemcpyDefault));
    }
    else
    {
      checkCudaErrors(cudaMemcpy(g0, g1, buf_size, cudaMemcpyDefault));
    }
  }

  checkCudaErrors(cudaEventRecord(stop_event, 0));
  checkCudaErrors(cudaEventSynchronize(stop_event));
  checkCudaErrors(cudaEventElapsedTime(&time_memcpy, start_event, stop_event));
  printf("cudaMemcpyPeer / cudaMemcpy between GPU%d and GPU%d: %.2fGB/s\n",
         gpuid[0],
         gpuid[1],
         (1.0f / (time_memcpy / 1000.0f)) * ((100.0f * buf_size)) / 1024.0f / 1024.0f / 1024.0f);

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and memcpy to GPU%d...\n", gpuid[0]);

  for (int i = 0; i < buf_size / sizeof(float); i++)
  {
    h0[i] = float(i % 4096);
  }

  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaMemcpy(g0, h0, buf_size, cudaMemcpyDefault));

  // Kernel launch configuration
  const dim3 threads(512, 1);
  const dim3 blocks((buf_size / sizeof(float)) / threads.x, 1);

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
  // output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         gpuid[1],
         gpuid[0],
         gpuid[1]);
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  SimpleKernel<<<blocks, threads>>>(g0, g1);

  checkCudaErrors(cudaDeviceSynchronize());

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
  // output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         gpuid[0],
         gpuid[1],
         gpuid[0]);
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  SimpleKernel<<<blocks, threads>>>(g1, g0);

  checkCudaErrors(cudaDeviceSynchronize());

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", gpuid[0]);
  checkCudaErrors(cudaMemcpy(h0, g0, buf_size, cudaMemcpyDefault));

  int error_count = 0;

  for (int i = 0; i < buf_size / sizeof(float); i++)
  {
    // Re-generate input data and apply 2x '* 2.0f' computation of both
    // kernel runs
    if (h0[i] != float(i % 4096) * 2.0f * 2.0f)
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0[i], (float(i % 4096) * 2.0f * 2.0f));

      if (error_count++ > 10)
      {
        break;
      }
    }
  }

  // Disable peer access (also unregisters memory for non-UVA cases)
  printf("Disabling peer access...\n");
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[1]));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaDeviceDisablePeerAccess(gpuid[0]));

  // Cleanup and shutdown
  printf("Shutting down...\n");
  checkCudaErrors(cudaEventDestroy(start_event));
  checkCudaErrors(cudaEventDestroy(stop_event));
  checkCudaErrors(cudaSetDevice(gpuid[0]));
  checkCudaErrors(cudaFree(g0));
  checkCudaErrors(cudaSetDevice(gpuid[1]));
  checkCudaErrors(cudaFree(g1));
  checkCudaErrors(cudaFreeHost(h0));

  for (int i = 0; i < gpu_n; i++)
  {
    checkCudaErrors(cudaSetDevice(i));
  }

  if (error_count != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }
  else
  {
    printf("Test passed\n");
    exit(EXIT_SUCCESS);
  }
}
