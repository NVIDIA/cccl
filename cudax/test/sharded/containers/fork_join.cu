//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief `sharded_array<T>::fork_from` / `join_into`: ordering declarations
 *        bridging a caller stream and the per-shard streams. Covers the eager
 *        producer -> fork -> per-shard consumers -> join -> reader chain with
 *        NO host synchronization between the stages, the same chain on an
 *        ADOPTED array over foreign streams, and the members inside a CUDA
 *        graph capture (record/wait become graph dependencies).
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
// Burn ~`cycles` GPU cycles so that a missing stream dependency surfaces as a
// stale read instead of accidental serialization.
__global__ void spin_kernel(long long cycles)
{
  const long long start = clock64();
  while (clock64() - start < cycles)
  {
  }
}

// Producer: value derived from the global index.
__global__ void produce_kernel(int* data, size_t n, size_t global_offset, int salt)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    data[i] = 2 * static_cast<int>(global_offset + i) + salt;
  }
}

// Consumer: out = in + 1.
__global__ void consume_kernel(const int* in, int* out, size_t n)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i] + 1;
  }
}

// In-place increment (graph relaunch check).
__global__ void increment_kernel(int* data, size_t n)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    data[i] += 1;
  }
}

constexpr int threads = 256;

inline unsigned int blocks_for(size_t n)
{
  return static_cast<unsigned int>((n + threads - 1) / threads);
}

// Producer on the caller stream -> fork_from -> per-shard consumers on the
// shard streams -> join_into -> reader (memcpy) on the caller stream. The
// ONLY host synchronization is the final caller-stream sync.
void test_eager_ordering(place_group& group)
{
  const size_t n = 1 << 20;
  auto in        = sharded_array<int>::allocate(group, n);
  auto out       = sharded_array<int>::allocate_like(in);

  // Sentinels, quiesced before the ordered chain starts.
  fill(group, in, -1);
  fill(group, out, -1);
  in.sync();
  out.sync();

  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));

  // Delay + produce on the caller stream (per shard: the producer writes
  // through each shard's pointer, all enqueued on the caller stream).
  spin_kernel<<<1, 1, 0, caller>>>(20'000'000);
  for (size_t i = 0; i < in.num_shards(); i++)
  {
    auto& s = in.shard(i);
    produce_kernel<<<blocks_for(s.size), threads, 0, caller>>>(s.data, s.size, s.global_offset, 7);
  }

  // Fork: shard streams now depend on the producer.
  in.fork_from(caller);

  // Per-shard consumers on the shard streams.
  in.each_shard->*[&out](size_t i, const auto& s) {
    consume_kernel<<<blocks_for(s.size), threads, 0, s.stream>>>(s.data, out.shard(i).data, s.size);
  };

  // Join: the caller stream now depends on every consumer.
  out.join_into(caller);

  ::std::vector<int> host(n, 0);
  for (size_t i = 0; i < out.num_shards(); i++)
  {
    const auto& s = out.shard(i);
    cuda_safe_call(cudaMemcpyAsync(host.data() + s.global_offset, s.data, s.size_bytes(), cudaMemcpyDefault, caller));
  }
  cuda_safe_call(cudaStreamSynchronize(caller)); // the only host sync

  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<int>(i) + 7 + 1);
  }

  cuda_safe_call(cudaStreamDestroy(caller));
}

// Same chain on an ADOPTED array: caller-owned device buffers and FOREIGN
// streams (created outside any place_group).
void test_adopted_foreign_streams()
{
  const size_t n_per = 1 << 19;
  const size_t parts = 2;
  const size_t n     = n_per * parts;

  cuda_safe_call(cudaSetDevice(0));

  ::std::vector<int*> buffers(parts, nullptr);
  ::std::vector<cudaStream_t> foreign(parts, nullptr);
  ::std::vector<shard<int>> shards(parts);
  for (size_t i = 0; i < parts; i++)
  {
    cuda_safe_call(cudaMalloc(&buffers[i], n_per * sizeof(int)));
    cuda_safe_call(cudaStreamCreate(&foreign[i]));
    shards[i].data          = buffers[i];
    shards[i].size          = n_per;
    shards[i].capacity      = n_per;
    shards[i].global_offset = i * n_per;
    shards[i].place         = data_place::device(0);
    shards[i].exec          = exec_place::device(0);
    shards[i].stream        = foreign[i];
  }

  auto data = sharded_array<int>::adopt(mv(shards));
  EXPECT(data.is_view());

  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));

  spin_kernel<<<1, 1, 0, caller>>>(20'000'000);
  for (size_t i = 0; i < data.num_shards(); i++)
  {
    auto& s = data.shard(i);
    produce_kernel<<<blocks_for(s.size), threads, 0, caller>>>(s.data, s.size, s.global_offset, 3);
  }

  data.fork_from(caller);

  data.each_shard->*[](const auto& s) {
    increment_kernel<<<blocks_for(s.size), threads, 0, s.stream>>>(s.data, s.size);
  };

  data.join_into(caller);

  ::std::vector<int> host(n, 0);
  for (size_t i = 0; i < data.num_shards(); i++)
  {
    const auto& s = data.shard(i);
    cuda_safe_call(cudaMemcpyAsync(host.data() + s.global_offset, s.data, s.size_bytes(), cudaMemcpyDefault, caller));
  }
  cuda_safe_call(cudaStreamSynchronize(caller)); // the only host sync

  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<int>(i) + 3 + 1);
  }

  cuda_safe_call(cudaStreamDestroy(caller));
  for (size_t i = 0; i < parts; i++)
  {
    cuda_safe_call(cudaStreamDestroy(foreign[i]));
    cuda_safe_call(cudaFree(buffers[i]));
  }
}

// fork_from/join_into INSIDE a CUDA graph capture: the record/wait pairs
// become graph dependencies; the instantiated graph replays the whole
// fork -> per-shard work -> join chain, repeatedly.
void test_capture(place_group& group)
{
  const size_t n = 1 << 20;
  auto data      = sharded_array<int>::allocate(group, n);

  iota(group, data, 0);
  data.sync();

  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));

  cuda_safe_call(cudaStreamBeginCapture(caller, cudaStreamCaptureModeGlobal));

  data.fork_from(caller);
  data.each_shard->*[](const auto& s) {
    increment_kernel<<<blocks_for(s.size), threads, 0, s.stream>>>(s.data, s.size);
  };
  data.join_into(caller);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(caller, &graph));
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  const int launches = 3;
  for (int r = 0; r < launches; r++)
  {
    cuda_safe_call(cudaGraphLaunch(exec, caller));
  }
  cuda_safe_call(cudaStreamSynchronize(caller));

  ::std::vector<int> host(n, 0);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<int>(i) + launches);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(caller));
}

// Degenerate inputs: empty containers and same-stream shards are no-ops.
void test_degenerate()
{
  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));

  sharded_array<int> empty;
  empty.fork_from(caller);
  empty.join_into(caller);

  {
    // Shards whose reference stream IS the caller stream: nothing to order.
    auto same = sharded_array<int>::allocate({{128, data_place::device(0), exec_place::device(0), caller}});
    same.fork_from(caller);
    same.join_into(caller);
    cuda_safe_call(cudaStreamSynchronize(caller));
  } // destroyed before its reference stream

  cuda_safe_call(cudaStreamDestroy(caller));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_eager_ordering(group);
  test_adopted_foreign_streams();
  test_capture(group);
  test_degenerate();

  return 0;
}
