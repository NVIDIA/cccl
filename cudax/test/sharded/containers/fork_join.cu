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
 *        ADOPTED array over foreign streams, adoption ONTO A LANE with the
 *        producer's timeline as `ready_on` (the fork folded into `adopt`,
 *        the join spelled with the same call-environment form), and the
 *        members inside a CUDA graph capture (record/wait become graph
 *        dependencies).
 */

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
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
  fill(in, -1);
  fill(out, -1);
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

// Adoption onto a LANE: caller-owned buffers (synchronous cudaMalloc) become a
// container whose reference streams are the group's lane streams, the
// producer's timeline enters once as `ready_on` (call-environment spelling),
// and the join back into the producer is the same spelling. The container
// knows its lane and its environments report it.
void test_adopt_on_lane(place_group& group)
{
  const size_t P     = group.size();
  const size_t n_per = 1 << 19;
  const size_t n     = n_per * P;

  ::std::vector<::std::pair<int*, size_t>> pieces(P);
  for (size_t i = 0; i < P; i++)
  {
    exec_place_scope scope(group.place(i));
    cuda_safe_call(cudaMalloc(&pieces[i].first, n_per * sizeof(int)));
    pieces[i].second = n_per;
  }

  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));
  const auto ce = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{caller}};

  // Producer on the caller stream, then adopt onto lane 1 depending on it:
  // the fork is part of adoption.
  spin_kernel<<<1, 1, 0, caller>>>(20'000'000);
  size_t off = 0;
  for (size_t i = 0; i < P; i++)
  {
    produce_kernel<<<blocks_for(n_per), threads, 0, caller>>>(pieces[i].first, n_per, off, 5);
    off += n_per;
  }
  auto data = sharded_array<int>::adopt(group.lane(1), pieces, ce);
  EXPECT(data.is_view());
  EXPECT(data.lane() == ::cuda::std::optional<size_t>{1});
  auto envs = default_envs(data);
  for (size_t i = 0; i < P; i++)
  {
    EXPECT(data.shard(i).stream == group.get_stream(i, 1));
    EXPECT(::cuda::experimental::places::query_lane_id(envs[i]) == ::cuda::std::optional<size_t>{1});
    EXPECT(data.shard(i).global_offset == i * n_per);
  }
  // A slice inherits the lane; a plain group means lane 0; adopted foreign
  // shards have none.
  EXPECT(data.slice(1, n - 1).lane() == ::cuda::std::optional<size_t>{1});
  EXPECT(sharded_array<int>::adopt(group, pieces).lane() == ::cuda::std::optional<size_t>{0});
  EXPECT(!sharded_array<int>::allocate_uniform(64, {0}).lane().has_value());

  // Lane-ordered consumers on the lane's streams, then join back into the
  // producer's timeline with the same call-environment spelling.
  data.each_shard->*[](const auto& s) {
    increment_kernel<<<blocks_for(s.size), threads, 0, s.stream>>>(s.data, s.size);
  };
  data.join_into(ce);

  ::std::vector<int> host(n, 0);
  for (size_t i = 0; i < P; i++)
  {
    const auto& s = data.shard(i);
    cuda_safe_call(cudaMemcpyAsync(host.data() + s.global_offset, s.data, s.size_bytes(), cudaMemcpyDefault, caller));
  }
  cuda_safe_call(cudaStreamSynchronize(caller)); // the only host sync
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<int>(i) + 5 + 1);
  }

  // Contract refusals: piece count must match the group; lane ids never wrap.
  bool threw = false;
  try
  {
    ::std::ignore = sharded_array<int>::adopt(group, ::std::vector<::std::pair<int*, size_t>>(P + 1));
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    ::std::ignore = group.lane(group.num_lanes());
  }
  catch (const ::std::out_of_range&)
  {
    threw = true;
  }
  EXPECT(threw);

  cuda_safe_call(cudaStreamDestroy(caller));
  for (size_t i = 0; i < P; i++)
  {
    cuda_safe_call(cudaFree(pieces[i].first));
  }
}

// fork_from/join_into INSIDE a CUDA graph capture: the record/wait pairs
// become graph dependencies; the instantiated graph replays the whole
// fork -> per-shard work -> join chain, repeatedly.
void test_capture(place_group& group)
{
  const size_t n = 1 << 20;
  auto data      = sharded_array<int>::allocate(group, n);

  iota(data, 0);
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

  auto group = place_group{make_locality_domain_grid()};

  test_eager_ordering(group);
  test_adopted_foreign_streams();
  test_adopt_on_lane(group);
  test_capture(group);
  test_degenerate();

  return 0;
}
