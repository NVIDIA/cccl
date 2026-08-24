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
 * @brief CUDA graph capture of the elementwise sharded pipeline: fork an
 *        origin stream to every shard stream (`fork_from`), record fill/transform/for_each
 *        with `blocking = false`, join back (`join_into`), instantiate, and replay —
 *        including replays with inputs mutated between launches, a
 *        cross-stream (different-color) captured dependency, and a check that
 *        the per-place SM confinement of the shard streams survives inside
 *        the instantiated graph.
 */

#include <cuda/stream>

#include <cuda/experimental/sharded.cuh>

#include <set>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place_scope;
using cuda::experimental::places::place_group;

namespace
{
struct scale_op
{
  __host__ __device__ float operator()(float x) const
  {
    return 2.0f * x;
  }
};

struct plus_half_op
{
  __host__ __device__ float operator()(float x) const
  {
    return x + 0.5f;
  }
};

struct bump_op
{
  __host__ __device__ void operator()(float& v, size_t) const
  {
    v += 1.0f;
  }
};

// The eager reference of the captured pipeline: out = 2 * in + 0.5 + 1
float expected(float in)
{
  return 2.0f * in + 1.5f;
}

__global__ void smid_probe_kernel(unsigned* smids)
{
  if (threadIdx.x == 0)
  {
    unsigned smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));
    smids[blockIdx.x] = smid;
  }
}

void test_pipeline_capture_and_replay(place_group& group)
{
  const size_t n = 1 << 20;
  auto in        = sharded_array<float>::allocate(group, n, /*color*/ 0);
  auto out       = sharded_array<float>::allocate(group, n, /*color*/ 0);

  fill(group, in, 1.0f); // eager warm-up outside capture (modules, pools)

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  // Capture the pipeline: fork, then transform(in -> out, x2),
  // transform(out += 0.5 in place), for_each(out += 1), then join
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  in.fork_from(origin);
  transform(group, in, out, scale_op{}, /*blocking=*/false);
  transform(group, out, plus_half_op{}, /*blocking=*/false);
  for_each(group, out, bump_op{}, /*blocking=*/false);
  out.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  EXPECT(graph != nullptr);

  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));

  // Replay with inputs mutated between launches (writes happen OUTSIDE the
  // graph; each launch recomputes from the current input values)
  ::std::vector<float> host(n);
  ::std::vector<float> input(n);
  for (int round = 0; round < 3; round++)
  {
    for (size_t i = 0; i < n; i++)
    {
      input[i] = static_cast<float>(round + 1) + static_cast<float>(i % 7);
    }
    in.copy_from_host(input.data());

    cuda_safe_call(cudaGraphLaunch(exec, origin));
    cuda_safe_call(cudaStreamSynchronize(origin));

    out.copy_to_host(host.data());
    for (size_t i = 0; i < n; i++)
    {
      EXPECT(host[i] == expected(input[i]));
    }
  }

  // Clobber the output and replay: the graph recomputes it
  fill(group, out, -1.0f);
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));
  out.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == expected(input[i]));
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}

// A captured cross-stream dependency: input and output allocated at different
// stream colors, so the out-of-place transform records event edges between
// two capturing shard streams.
void test_cross_color_dependency(place_group& group)
{
  if (group.num_stream_colors() < 2)
  {
    return;
  }

  const size_t n = 100003;
  auto in        = sharded_array<float>::allocate(group, n, /*color*/ 0);
  auto out       = sharded_array<float>::allocate(group, n, /*color*/ 1);

  fill(group, in, 4.0f);

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));

  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  in.fork_from(origin);
  out.fork_from(origin);
  transform(group, in, out, scale_op{}, /*blocking=*/false); // records in->out event edges
  out.join_into(origin);
  in.join_into(origin);

  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  ::std::vector<float> host(n);
  out.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 8.0f);
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
}

// The green-context SM confinement of the shard streams must survive inside
// the instantiated graph: kernels captured from two locality-domain streams
// must land on disjoint SM sets when replayed.
void test_confinement_in_graph(place_group& group)
{
  if (group.size() < 2)
  {
    return; // needs at least two places to compare SM sets
  }
  // Only meaningful when the places are domains of ONE device (disjoint SM
  // partitions); distinct whole devices trivially pass, which is fine too.

  const size_t num_places = group.size();
  const int blocks        = 64;

  ::std::vector<unsigned*> smid_bufs(num_places);
  for (size_t i = 0; i < num_places; i++)
  {
    exec_place_scope scope(group.place(i));
    cuda_safe_call(cudaMalloc(&smid_bufs[i], blocks * sizeof(unsigned)));
  }

  // Warm-up launch outside capture on each place stream
  for (size_t i = 0; i < num_places; i++)
  {
    exec_place_scope scope(group.place(i));
    smid_probe_kernel<<<blocks, 32, 0, group.get_stream(i, 0)>>>(smid_bufs[i]);
    cuda_safe_call(cudaGetLastError());
  }
  group.sync();

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  cuda_safe_call(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
  for (size_t i = 0; i < num_places; i++)
  {
    ::cuda::stream_ref{group.get_stream(i, 0)}.wait(::cuda::stream_ref{origin});
    exec_place_scope scope(group.place(i));
    smid_probe_kernel<<<blocks, 32, 0, group.get_stream(i, 0)>>>(smid_bufs[i]);
    cuda_safe_call(cudaGetLastError());
    ::cuda::stream_ref{origin}.wait(::cuda::stream_ref{group.get_stream(i, 0)});
  }
  cudaGraph_t graph = nullptr;
  cuda_safe_call(cudaStreamEndCapture(origin, &graph));
  cudaGraphExec_t exec = nullptr;
  cuda_safe_call(cudaGraphInstantiate(&exec, graph, 0));
  cuda_safe_call(cudaGraphLaunch(exec, origin));
  cuda_safe_call(cudaStreamSynchronize(origin));

  ::std::vector<::std::set<unsigned>> smid_sets(num_places);
  for (size_t i = 0; i < num_places; i++)
  {
    ::std::vector<unsigned> h(blocks);
    cuda_safe_call(cudaMemcpy(h.data(), smid_bufs[i], blocks * sizeof(unsigned), cudaMemcpyDefault));
    smid_sets[i].insert(h.begin(), h.end());
  }
  for (size_t i = 0; i < num_places; i++)
  {
    for (size_t j = i + 1; j < num_places; j++)
    {
      if (device_ordinal(group.place(i).affine_data_place()) != device_ordinal(group.place(j).affine_data_place()))
      {
        continue; // different devices: SMID spaces are unrelated
      }
      for (unsigned smid : smid_sets[i])
      {
        EXPECT(smid_sets[j].count(smid) == 0);
      }
    }
  }

  cuda_safe_call(cudaGraphExecDestroy(exec));
  cuda_safe_call(cudaGraphDestroy(graph));
  cuda_safe_call(cudaStreamDestroy(origin));
  for (size_t i = 0; i < num_places; i++)
  {
    exec_place_scope scope(group.place(i));
    cuda_safe_call(cudaFree(smid_bufs[i]));
  }
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_pipeline_capture_and_replay(group);
  test_cross_color_dependency(group);
  test_confinement_in_graph(group);

  return 0;
}
