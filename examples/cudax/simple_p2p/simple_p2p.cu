/*
 * This sample demonstrates a combination of Peer-to-Peer (P2P) and
 * Unified Virtual Address Space (UVA) features new to SDK 4.0
 */

// includes, system
#include <cuda/memory_resource>

#include <cuda/experimental/algorithm.cuh>
#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/device.cuh>
#include <cuda/experimental/launch.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <algorithm>

#include <stdio.h>
#include <stdlib.h>

namespace cudax = cuda::experimental;

#define checkCudaErrors

struct SimpleKernel
{
  template <typename Dimensions>
  __device__ void operator()(Dimensions dims, ::cuda::std::span<const float> src, ::cuda::std::span<float> dst)
  {
    // Just a dummy kernel, doing enough for us to verify that everything
    // worked
    const auto idx = dims.rank(cudax::thread);
    dst[idx]       = src[idx] * 2.0f;
  }
};

std::vector<cudax::device_ref> find_peers_group()
{
  // Check possibility for peer access
  printf("\nChecking GPU(s) for support of peer to peer memory access...\n");

  std::vector<cudax::device_ref> peers;
  for (auto& dev_i : cudax::devices)
  {
    for (auto& dev_j : cudax::devices)
    {
      if (dev_i != dev_j)
      {
        bool can_access_peer = dev_i.is_peer_accessible_from(dev_j);
        // Save all peers of a first device found with a peer
        if (can_access_peer && peers.size() == 0)
        {
          peers = dev_i.get_peers();
          peers.push_back(dev_i);
        }
        printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
               dev_i.get_name().c_str(),
               dev_i.get(),
               dev_j.get_name().c_str(),
               dev_j.get(),
               can_access_peer ? "Yes" : "No");
      }
    }
  }

  if (peers.size() == 0)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required, waving the test.\n");
    exit(2);
  }

  return peers;
}

template <typename BufferType>
void benchmark_cross_device_ping_pong_copy(
  cudax::stream_ref dev0_stream, cudax::stream_ref dev1_stream, BufferType& dev0_buffer, BufferType& dev1_buffer)
{
  constexpr int cpy_count = 100;
  auto start_event        = dev0_stream.record_timed_event();
  for (int i = 0; i < cpy_count; i++)
  {
    // Ping-pong copy between GPUs
    if (i % 2 == 0)
    {
      cudax::copy_bytes(dev0_stream, dev0_buffer, dev1_buffer);
    }
    else
    {
      cudax::copy_bytes(dev0_stream, dev1_buffer, dev0_buffer);
    }
  }

  auto end_event = dev0_stream.record_timed_event();
  dev0_stream.wait();
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
  cudax::device_ref dev0 = dev0_stream.device().get();
  cudax::device_ref dev1 = dev1_stream.device().get();

  // Prepare host buffer and copy to GPU 0
  printf("Preparing host buffer and copy to GPU%d...\n", dev0.get());

  // This will be a pinned memory vector once available
  cudax::uninitialized_buffer<float, cuda::mr::host_accessible> host_buffer(
    cuda::mr::pinned_memory_resource(), dev0_buffer.size());
  std::generate(host_buffer.begin(), host_buffer.end(), []() {
    static int i = 0;
    return (i++) % 4096;
  });

  cudax::copy_bytes(dev0_stream, host_buffer, dev0_buffer);
  dev1_stream.wait(dev0_stream);

  // Kernel launch configuration
  auto dims = cudax::distribute<512>(dev0_buffer.size());

  // Run kernel on GPU 1, reading input from the GPU 0 buffer, writing
  // output to the GPU 1 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         dev1.get(),
         dev0.get(),
         dev1.get());
  cudax::launch(dev1_stream, dims, SimpleKernel{}, dev0_buffer, dev1_buffer);
  dev0_stream.wait(dev1_stream);

  // Run kernel on GPU 0, reading input from the GPU 1 buffer, writing
  // output to the GPU 0 buffer
  printf("Run kernel on GPU%d, taking source data from GPU%d and writing to "
         "GPU%d...\n",
         dev0.get(),
         dev1.get(),
         dev0.get());
  cudax::launch(dev0_stream, dims, SimpleKernel{}, dev1_buffer, dev0_buffer);

  // Copy data back to host and verify
  printf("Copy data back to host from GPU%d and verify results...\n", dev0.get());
  cudax::copy_bytes(dev0_stream, dev0_buffer, host_buffer);
  dev0_stream.wait();

  int error_count = 0;
  for (int i = 0; i < host_buffer.size(); i++)
  {
    cuda::std::span host_span(host_buffer);
    // Re-generate input data and apply 2x '* 2.0f' computation of both
    // kernel runs
    float expected = float(i % 4096) * 2.0f * 2.0f;
    if (host_span[i] != expected)
    {
      printf("Verification error @ element %i: val = %f, ref = %f\n", i, host_span[i], expected);

      if (error_count++ > 10)
      {
        break;
      }
    }
  }
}

int main(int argc, char** argv)
{
  printf("[%s] - Starting...\n", argv[0]);

  // Number of GPUs
  printf("Checking for multiple GPUs...\n");
  printf("CUDA-capable device count: %lu\n", cudax::devices.size());

  if (cudax::devices.size() < 2)
  {
    printf("Two or more GPUs with Peer-to-Peer access capability are required for "
           "%s.\n",
           argv[0]);
    printf("Waiving test.\n");
    exit(2);
  }

  auto peers = find_peers_group();

  cudax::stream dev0_stream(peers[0]);
  cudax::stream dev1_stream(peers[1]);

  printf("Enabling peer access between GPU%d and GPU%d...\n", peers[0].get(), peers[1].get());
  cudax::mr::async_memory_resource dev0_resource(peers[0]);
  dev0_resource.enable_peer_access(peers[1]);
  cudax::mr::async_memory_resource dev1_resource(peers[1]);
  dev1_resource.enable_peer_access(peers[0]);

  // Allocate buffers
  constexpr size_t buf_cnt = 1024 * 1024 * 256;
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
  dev0_resource.disable_peer_access(peers[1]);
  dev1_resource.disable_peer_access(peers[0]);

  // No cleanup needed
}
