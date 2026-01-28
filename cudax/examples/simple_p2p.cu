/* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features.
 */

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/memory_resource>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

namespace cudax = cuda::experimental;

struct simple_kernel
{
  template <typename Configuration>
  __device__ void operator()(Configuration config, ::cuda::std::span<const float> src, ::cuda::std::span<float> dst)
  {
    // Just a dummy kernel, doing enough for us to verify that everything worked
    const auto idx = cuda::gpu_thread.rank(cuda::grid, config);
    dst[idx]       = src[idx] * 2.0f;
  }
};

void print_peer_accessibility()
{
  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  for (auto& dev_i : cuda::devices)
  {
    for (auto& dev_j : cuda::devices)
    {
      if (dev_i != dev_j)
      {
        bool can_access_peer = dev_i.has_peer_access_to(dev_j);
        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
               dev_i.name().data(),
               dev_i.get(),
               dev_j.name().data(),
               dev_j.get(),
               can_access_peer ? "Yes" : "No");
      }
    }
  }
}

template <typename BufferType>
void benchmark_cross_device_ping_pong_copy(
  cudax::stream_ref dev0_stream, cudax::stream_ref dev1_stream, BufferType& dev0_buffer, BufferType& dev1_buffer)
{
  // Use dev1 stream due to some surprising performance issue
  constexpr int cpy_count = 100;
  auto start_event        = dev1_stream.record_timed_event();
  for (int i = 0; i < cpy_count; i++)
  {
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      cuda::copy_bytes(dev1_stream, dev0_buffer, dev1_buffer);
    }
    else
    {
      cuda::copy_bytes(dev1_stream, dev1_buffer, dev0_buffer);
    }
  }

  auto end_event = dev1_stream.record_timed_event();
  dev1_stream.sync();
  cuda::std::chrono::duration<double> duration(end_event - start_event);
  printf("Peer copy between GPU%d and GPU%d: %.2fGB/s\n",
         dev0_stream.device().get(),
         dev1_stream.device().get(),
         (static_cast<float>(cpy_count * dev0_buffer.size_bytes()) / (1024 * 1024 * 1024) / duration.count()));
}

template <typename BufferType>
void test_cross_device_access_from_kernel(
  cudax::stream_ref dev0_stream, cudax::stream_ref dev1_stream, BufferType& dev0_buffer, BufferType& dev1_buffer)
{
  cuda::device_ref dev0 = dev0_stream.device();
  cuda::device_ref dev1 = dev1_stream.device();

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and copy to GPU%d...\n", dev0.get());

  // This will be a pinned memory vector once available
  cudax::uninitialized_buffer<float, cuda::mr::host_accessible> host_buffer(
    cuda::mr::legacy_pinned_memory_resource(), dev0_buffer.size());
  std::generate(host_buffer.begin(), host_buffer.end(), []() {
    static int i = 0;
    return static_cast<float>((i++) % 4096);
  });

  cuda::copy_bytes(dev0_stream, host_buffer, dev0_buffer);
  dev1_stream.wait(dev0_stream);

  // Kernel launch configuration
  auto config = cuda::distribute<512>(dev0_buffer.size());

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         dev1.get(),
         dev0.get(),
         dev1.get());
  cudax::launch(dev1_stream, config, simple_kernel{}, dev0_buffer, dev1_buffer);
  dev0_stream.wait(dev1_stream);

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         dev0.get(),
         dev1.get(),
         dev0.get());
  cudax::launch(dev0_stream, config, simple_kernel{}, dev1_buffer, dev0_buffer);

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", dev0.get());
  cuda::copy_bytes(dev0_stream, dev0_buffer, host_buffer);
  dev0_stream.sync();

  int error_count = 0;
  for (size_t i = 0; i < host_buffer.size(); i++)
  {
    cuda::std::span<float> host_span(host_buffer);
    // Re-generate input data and apply 2x '* 2.0f' computation of both kernel runs
    float expected = float(i % 4096) * 2.0f * 2.0f;
    if (host_span[i] != expected)
    {
      printf("Verification error @ element %zu: val = %f, ref = %f\n", i, host_span[i], expected);

      if (error_count++ > 10)
      {
        break;
      }
    }
  }
  if (error_count != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }
}

int main([[maybe_unused]] int argc, char** argv)
try
{
  printf("[%s] - Starting...\n", argv[0]);

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  printf("CUDA-capable device count: %zu\n", cuda::devices.size());

  if (cuda::devices.size() < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for %s.\n", argv[0]);
    printf("Waiving test.\n");
    return 0;
  }

  // Print full peer access matrix
  print_peer_accessibility();

  // But use a shorthand to find all peers of a device
  std::vector<cuda::device_ref> peers;
  for (auto& dev : cuda::devices)
  {
    const auto dev_peers = dev.peers();
    if (dev_peers.size() != 0)
    {
      peers.assign(dev_peers.begin(), dev_peers.end());
      peers.insert(peers.begin(), dev);
      break;
    }
  }

  if (peers.size() == 0)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required, waving the test.\n");
    return 0;
  }

  cuda::stream dev0_stream(peers[0]);
  cuda::stream dev1_stream(peers[1]);

  printf("Enabling peer access between GPU%d and GPU%d...\n", peers[0].get(), peers[1].get());
  cuda::device_memory_pool_ref dev0_resource = cuda::device_default_memory_pool(peers[0]);
  dev0_resource.enable_access_from(peers[1]);
  cuda::device_memory_pool_ref dev1_resource = cuda::device_default_memory_pool(peers[1]);
  dev1_resource.enable_access_from(peers[0]);

  // Allocate buffers
  constexpr size_t buf_cnt = 1024 * 1024 * 16;
  printf("Allocating buffers (%iMB on GPU%d, GPU%d and CPU Host)...\n",
         int(buf_cnt / 1024 / 1024 * sizeof(float)),
         peers[0].get(),
         peers[1].get());

  cudax::uninitialized_buffer<float, cuda::mr::device_accessible> dev0_buffer(dev0_resource, buf_cnt);
  cudax::uninitialized_buffer<float, cuda::mr::device_accessible> dev1_buffer(dev1_resource, buf_cnt);

  benchmark_cross_device_ping_pong_copy(dev0_stream, dev1_stream, dev0_buffer, dev1_buffer);

  test_cross_device_access_from_kernel(dev0_stream, dev1_stream, dev0_buffer, dev1_buffer);

  // Disable peer access
  printf("Disabling peer access...\n");
  dev0_resource.disable_access_from(peers[1]);
  dev1_resource.disable_access_from(peers[0]);

  // No cleanup needed
  printf("Test passed\n");
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
