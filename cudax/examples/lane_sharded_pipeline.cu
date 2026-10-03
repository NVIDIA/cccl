//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Sharded algorithms as senders, on lane schedulers.
//
// A `lane_scheduler` is one CUDA stream. A sharded array is N shards, each a
// span plus the lane it lives on. This example writes two algorithms over such
// an array, `transform` and `reduce`, as senders, and runs three transforms
// back to back followed by a reduce:
//
//     x --*2--> y --+1--> x --*3--> y --sum--> result
//
// The composition rule is the point of the example. A verb takes and returns a
// *bundle* of per-shard senders, one per lane (`start(x) | transform(...) | ...`). Elementwise verbs map over the
// bundle shard by shard, so a chain of transforms is one stream-ordered chain
// per lane and needs no synchronization at all. Only `reduce` brings the lanes
// together: a `when_all` over the bundle (the fork, one event recorded on the
// lane the pipeline started on, before any shard starts) and a
// `continues_on(lane 0)` (the join, one event per other lane). For the whole
// pipeline, that is N-1 fork waits and N-1 join waits, and nothing else.
//
// Run with `--graph` to capture the pipeline into a CUDA graph instead and
// write it as `lane_sharded_pipeline.dot`: N independent chains out of one root,
// joined once at the sum. (`dot -Tpdf lane_sharded_pipeline.dot -o pipeline.pdf`)

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/execution.cuh>

#include <cstdio>
#include <cstring>
#include <exception>

namespace ex = cuda::experimental::execution;

void check(cudaError_t st, const char* what)
{
  if (st != cudaSuccess)
  {
    throw cuda::cuda_error(st, what);
  }
}

// ----------------------------------------------------------------------------
// A sharded array: N shards, each a span and the lane it lives on.
template <size_t N>
struct sharded_view
{
  int* data[N];
  int shard_size;
  ex::lane_scheduler lane[N];
};

// A bundle of per-shard senders, one per lane. This is what verbs take and
// return. Combining a bundle shard-wise is `map`; a `when_all` over it is the
// only way lanes meet.
template <class... Senders>
struct bundle
{
  cuda::std::tuple<Senders...> shards;
};
template <class... Senders>
bundle(cuda::std::tuple<Senders...>) -> bundle<Senders...>;

template <class Tuple, class Fn, size_t... I>
auto map(Tuple&& t, Fn&& fn, cuda::std::index_sequence<I...>)
{
  return bundle{cuda::std::make_tuple(fn(cuda::std::get<I>(static_cast<Tuple&&>(t)), I)...)};
}

// ----------------------------------------------------------------------------
// The verbs.

// start(view): begin on every shard's lane.
template <size_t N, size_t... I>
auto start(const sharded_view<N>& v, cuda::std::index_sequence<I...>)
{
  return bundle{cuda::std::make_tuple(ex::schedule(v.lane[I])...)};
}
template <size_t N>
auto start(const sharded_view<N>& v)
{
  return start(v, cuda::std::make_index_sequence<N>{});
}

// The environment a CUB call needs to run on a lane: its stream, and a memory
// resource for CUB's own scratch storage.
template <class Mr>
auto cub_env(const ex::lane_scheduler& lane, Mr mr)
{
  return cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream, lane.query(cuda::get_stream)},
                                   cuda::std::execution::prop{cuda::mr::get_memory_resource, mr}};
}

// Verbs are closures: `bundle | verb(...)` applies the verb to every shard.
template <class Fn>
struct verb
{
  Fn fn;
  template <class... Senders>
  friend auto operator|(bundle<Senders...> b, verb v)
  {
    return v.fn(std::move(b));
  }
};
template <class Fn>
verb(Fn) -> verb<Fn>;

// transform(in, out, op): `out[k] = op(in[k])` on shard k's lane. Elementwise:
// maps over the bundle, no lane meets another.
template <size_t N, class Op, class Mr>
auto transform(const sharded_view<N>& in, const sharded_view<N>& out, Op op, Mr mr)
{
  return verb{[=](auto b) {
    static_assert(cuda::std::tuple_size_v<decltype(b.shards)> == N, "one sender per shard");
    return map(
      std::move(b.shards),
      [=](auto shard, size_t k) {
        return std::move(shard) | ex::then([=] {
                 check(cub::DeviceTransform::Transform(
                         cuda::std::make_tuple(in.data[k]), out.data[k], in.shard_size, op, cub_env(in.lane[k], mr)),
                       "DeviceTransform::Transform");
               });
      },
      cuda::std::make_index_sequence<N>{});
  }};
}

__global__ void sum_partials(const int* partials, int n, int* result)
{
  int acc = 0;
  for (int i = 0; i < n; ++i)
  {
    acc += partials[i];
  }
  *result = acc;
}

// reduce(in, partials, result): each shard reduces into its own partial on its
// own lane; then the lanes meet once, on lane 0, where the partials are added.
template <class Bundle, size_t N, class Mr, size_t... I>
auto reduce_impl(Bundle b, const sharded_view<N>& in, int* partials, int* result, Mr mr, cuda::std::index_sequence<I...>)
{
  auto shard_reduce = [=](auto shard, size_t k) {
    return std::move(shard) | ex::then([=] {
             check(cub::DeviceReduce::Reduce(
                     in.data[k], partials + k, in.shard_size, cuda::std::plus<>{}, 0, cub_env(in.lane[k], mr)),
                   "DeviceReduce::Reduce");
           });
  };
  return ex::when_all(shard_reduce(cuda::std::get<I>(std::move(b.shards)), I)...) // the fork
       | ex::continues_on(in.lane[0]) // the join
       | ex::then([=] {
           sum_partials<<<1, 1, 0, in.lane[0].stream()>>>(partials, static_cast<int>(N), result);
         });
}
template <size_t N, class Mr>
auto reduce(const sharded_view<N>& in, int* partials, int* result, Mr mr)
{
  return verb{[=](auto b) {
    return reduce_impl(std::move(b), in, partials, result, mr, cuda::std::make_index_sequence<N>{});
  }};
}

// ----------------------------------------------------------------------------
struct times2
{
  __host__ __device__ int operator()(int x) const
  {
    return 2 * x;
  }
};
struct plus1
{
  __host__ __device__ int operator()(int x) const
  {
    return x + 1;
  }
};
struct times3
{
  __host__ __device__ int operator()(int x) const
  {
    return 3 * x;
  }
};

__global__ void fill(int* p, int n, int v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    p[i] = v;
  }
}

// A receiver for running a sender by hand (needed under stream capture, where
// sync_wait must not be used: it would synchronize inside the capture).
struct no_op_receiver
{
  using receiver_concept = ex::receiver_t;
  void set_value() noexcept {}
  template <class E>
  void set_error(E&&) noexcept
  {}
  void set_stopped() noexcept {}
};

int main(int argc, char** argv)
{
  const bool as_graph = argc > 1 && std::strcmp(argv[1], "--graph") == 0;
  try
  {
    constexpr size_t N   = 3;
    const int shard_size = 1 << 18;
    const int n          = shard_size * N;
    cuda::device_ref dev{0};
    cuda::device_memory_pool_ref mr = cuda::device_default_memory_pool(dev);

    // One lane per shard. The lanes are plain streams; the scheduler stores
    // nothing else.
    cuda::stream streams[N] = {cuda::stream{dev}, cuda::stream{dev}, cuda::stream{dev}};
    ex::lane_scheduler lanes[N];
    for (size_t k = 0; k < N; ++k)
    {
      lanes[k] = ex::lane_scheduler{streams[k]};
    }

    // Two sharded arrays x and y, shard k of each on lane k, and the partials.
    cuda::device_buffer<int> xbuf{streams[0], mr, static_cast<size_t>(n), cuda::no_init};
    cuda::device_buffer<int> ybuf{streams[0], mr, static_cast<size_t>(n), cuda::no_init};
    cuda::device_buffer<int> partials{streams[0], mr, N, cuda::no_init};
    cuda::device_buffer<int> result{streams[0], mr, 1, cuda::no_init};
    sharded_view<N> x{{xbuf.data(), xbuf.data() + shard_size, xbuf.data() + 2 * shard_size},
                      shard_size,
                      {lanes[0], lanes[1], lanes[2]}};
    sharded_view<N> y{{ybuf.data(), ybuf.data() + shard_size, ybuf.data() + 2 * shard_size},
                      shard_size,
                      {lanes[0], lanes[1], lanes[2]}};
    fill<<<(n + 255) / 256, 256, 0, streams[0].get()>>>(xbuf.data(), n, 1);
    streams[0].sync();

    // The pipeline. It begins on lane 0, which is where the reduce's when_all
    // forks the other lanes from.
    auto pipeline = ex::schedule(lanes[0]) | ex::let_value([&] {
                      return start(x) //
                           | transform(x, y, times2{}, mr) //
                           | transform(y, x, plus1{}, mr) //
                           | transform(x, y, times3{}, mr) //
                           | reduce(y, partials.data(), result.data(), mr);
                    });

    const int expected = 3 * (2 * 1 + 1) * n; // 9 per element

    if (!as_graph)
    {
      // Eager: every `then` body above runs now, on the host, and enqueues onto
      // its lane. sync_wait returns when every lane is done.
      ex::sync_wait(std::move(pipeline));
    }
    else
    {
      // Captured: the same chain, enqueued into a capture that starts on lane 0.
      // The fork events bring the other lanes into the capture; the join events
      // become graph edges.
      cudaGraph_t graph{};
      check(cudaStreamBeginCapture(streams[0].get(), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
      {
        auto op = ex::connect(std::move(pipeline), no_op_receiver{});
        ex::start(op);
      }
      // Every lane's tail must be joined back into the capturing stream before
      // the capture ends. The pipeline's own join covers the reduce; CUB's
      // scratch frees were enqueued on each lane after it.
      for (size_t k = 1; k < N; ++k)
      {
        streams[0].wait(streams[k]);
      }
      check(cudaStreamEndCapture(streams[0].get(), &graph), "cudaStreamEndCapture");
      size_t nodes = 0, edges = 0;
      check(cudaGraphGetNodes(graph, nullptr, &nodes), "cudaGraphGetNodes");
      check(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges), "cudaGraphGetEdges");
      check(cudaGraphDebugDotPrint(graph, "lane_sharded_pipeline.dot", 0), "cudaGraphDebugDotPrint");
      std::printf("captured graph: %zu nodes, %zu edges (lane_sharded_pipeline.dot)\n", nodes, edges);

      cudaGraphExec_t exec{};
      check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
      check(cudaGraphLaunch(exec, streams[0].get()), "cudaGraphLaunch");
      streams[0].sync();
      cudaGraphExecDestroy(exec);
      cudaGraphDestroy(graph);
    }

    int r = 0;
    check(cudaMemcpy(&r, result.data(), sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    std::printf(
      "%s: sum over %d elements = %d (%s)\n", as_graph ? "graph" : "eager", n, r, r == expected ? "ok" : "WRONG");
    return r == expected ? 0 : 1;
  }
  catch (cuda::cuda_error const& e)
  {
    std::printf("CUDA error: %s\n", e.what());
  }
  catch (std::exception const& e)
  {
    std::printf("Exception: %s\n", e.what());
  }
  return 1;
}
