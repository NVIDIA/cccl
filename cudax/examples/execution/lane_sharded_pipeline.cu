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
// an array, `transform` and `reduce`, as senders, and runs a transform followed
// by a reduce:
//
//     start(x) | transform(x, y, *2) | reduce(y, result)
//
// Verbs take only data. The memory resource for scratch comes from the sender
// environment, read once at the root of the pipeline; `start` puts it and the
// shard's lane into a per-shard context that the bundle's values carry through
// the verbs. The caller supplies the resource once, as the environment of
// sync_wait (or of the receiver, under capture).
//
// The composition rule is the point of the example. A verb takes and returns a
// *bundle* of per-shard senders, one per lane (`start(x) | transform(...) | ...`).
// Elementwise verbs map over the bundle shard by shard, so a chain of transforms
// is one stream-ordered chain per lane and needs no synchronization at all. Only
// `reduce` brings the lanes together: a `when_all` over the bundle (the fork, one
// event recorded on the lane the pipeline started on, before any shard starts)
// and a `continues_on(lane 0)` (the join, one event per other lane). For the
// whole pipeline, that is N-1 fork waits and N-1 join waits, and nothing else.
//
// The reduce's partials are its own scratch: a scoped allocation, a sender that
// allocates on the lane the reduce joins on, held by a `let_value` scope inside
// the verb and freed when the reduce is done. The caller never sees them.
//
// Why only one transform: reading the environment at the root makes everything
// below it environment-dependent, and nvcc's device front end (cicc) re-derives
// the dependent sender machinery at every nesting level. Measured on this file
// with 3 shards: 0/1/2/3 transforms -> cicc 8 s / 10 s / 77 s / 785 s. The
// three-transform version, with a reproducer script and the measurements, is on
// branch senders/lane-scheduler-slow-compile-repro.
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
// Scoped allocation as a sender.
//
// `allocate_on<T>(n)` completes with a buffer of n elements allocated,
// stream-ordered, on the lane the sender runs on, through the environment's
// memory resource. Held by a `let_value` scope,
// the buffer lives in the operation state for the inner work and is freed on
// the same lane when the scope ends. The lane it is allocated on must be ordered
// after all of its readers: for data shared by every shard, that is the lane the
// pipeline joins on.
//
// cudax operation states are host/device, and nvcc's execution-space check
// rejects `cuda::device_buffer`'s host-only destructor reached from one; the
// buffer travels in this thin wrapper whose move and destructor are exempted,
// as cudax does for its own tuples.
template <class T>
struct scoped_buffer
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit scoped_buffer(cuda::device_buffer<T>&& b)
      : buf_(std::move(b))
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE scoped_buffer(scoped_buffer&& o) noexcept
      : buf_(std::move(o.buf_))
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE ~scoped_buffer() {}
  T* data()
  {
    return buf_.data();
  }

private:
  cuda::device_buffer<T> buf_;
};

template <class T>
auto allocate_on(size_t n)
{
  return ex::when_all(ex::read_env(ex::get_scheduler), ex::read_env(cuda::mr::get_memory_resource))
       | ex::then([=](auto lane, auto mr) {
           return scoped_buffer<T>{cuda::device_buffer<T>{lane.query(cuda::get_stream), mr, n, cuda::no_init}};
         });
}

// What a verb body needs on a shard's lane: the lane itself (its stream) and a
// memory resource for scratch storage. The pipeline reads the resource from the
// environment once, at its root (a forwarding query, so the caller supplies it
// once, as the environment of sync_wait or of the receiver under capture);
// `start` puts it in every shard's context, and every shard sender from then on
// completes with that context: the bundle's values *are* the per-shard context,
// and verbs are plain `then`s that pass it along.
//
// (Reading the resource with `read_env` inside each shard's chain works too, but
// makes every downstream sender environment-dependent, which cudax re-derives at
// each nesting level: the compile time of this file went from about a minute to
// well over ten. Read the environment once, at the root.)
template <class Mr>
struct shard_ctx
{
  ex::lane_scheduler lane;
  Mr mr;

  // The environment a CUB call needs: the lane's stream, and the resource for
  // CUB's own scratch.
  auto cub_env() const
  {
    return cuda::std::execution::env{cuda::std::execution::prop{cuda::get_stream, lane.query(cuda::get_stream)},
                                     cuda::std::execution::prop{cuda::mr::get_memory_resource, mr}};
  }
};

// ----------------------------------------------------------------------------
// The verbs.

// start(view, mr): begin on every shard's lane, completing with the shard's
// context.
template <size_t N, class Mr, size_t... I>
auto start(const sharded_view<N>& v, Mr mr, cuda::std::index_sequence<I...>)
{
  auto on_lane = [mr](ex::lane_scheduler lane) {
    return ex::schedule(lane) | ex::then([=] {
             return shard_ctx<Mr>{lane, mr};
           });
  };
  return bundle{cuda::std::make_tuple(on_lane(v.lane[I])...)};
}
template <size_t N, class Mr>
auto start(const sharded_view<N>& v, Mr mr)
{
  return start(v, mr, cuda::std::make_index_sequence<N>{});
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
template <size_t N, class Op>
auto transform(const sharded_view<N>& in, const sharded_view<N>& out, Op op)
{
  return verb{[=](auto b) {
    static_assert(cuda::std::tuple_size_v<decltype(b.shards)> == N, "one sender per shard");
    return map(
      std::move(b.shards),
      [=](auto shard, size_t k) {
        return std::move(shard) | ex::then([=](auto ctx) {
                 check(cub::DeviceTransform::Transform(
                         cuda::std::make_tuple(in.data[k]), out.data[k], in.shard_size, op, ctx.cub_env()),
                       "DeviceTransform::Transform");
                 return ctx;
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

// reduce(in, result): each shard reduces into its own partial on its own lane;
// then the lanes meet once, on lane 0, where the partials are added into
// `result`. The partials are the reduce's own scratch: a scoped allocation on
// lane 0 -- the join target, so that their free is ordered after the sum that
// reads them -- held for exactly the duration of the reduce.
template <class Bundle, size_t N, size_t... I>
auto reduce_impl(Bundle b, const sharded_view<N>& in, int* result, cuda::std::index_sequence<I...>)
{
  return allocate_on<int>(N) //
       | ex::let_value([b = std::move(b), in, result](scoped_buffer<int>& partials) mutable {
           auto shard_reduce = [=, p = partials.data()](auto shard, size_t k) {
             return std::move(shard) | ex::then([=](auto ctx) {
                      check(cub::DeviceReduce::Reduce(
                              in.data[k], p + k, in.shard_size, cuda::std::plus<>{}, 0, ctx.cub_env()),
                            "DeviceReduce::Reduce");
                    });
           };
           return ex::when_all(shard_reduce(cuda::std::get<I>(std::move(b.shards)), I)...) // the fork
                | ex::continues_on(in.lane[0]) // the join
                | ex::then([=, p = partials.data()] {
                    sum_partials<<<1, 1, 0, in.lane[0].stream()>>>(p, static_cast<int>(N), result);
                  });
         });
}
template <size_t N>
auto reduce(const sharded_view<N>& in, int* result)
{
  return verb{[=](auto b) {
    return reduce_impl(std::move(b), in, result, cuda::std::make_index_sequence<N>{});
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

__global__ void fill(int* p, int n, int v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    p[i] = v;
  }
}

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

    // Two sharded arrays x and y, shard k of each on lane k, and the result.
    cuda::device_buffer<int> xbuf{streams[0], mr, static_cast<size_t>(n), cuda::no_init};
    cuda::device_buffer<int> ybuf{streams[0], mr, static_cast<size_t>(n), cuda::no_init};
    cuda::device_buffer<int> result{streams[0], mr, 1, cuda::no_init};
    sharded_view<N> x{{xbuf.data(), xbuf.data() + shard_size, xbuf.data() + 2 * shard_size},
                      shard_size,
                      {lanes[0], lanes[1], lanes[2]}};
    sharded_view<N> y{{ybuf.data(), ybuf.data() + shard_size, ybuf.data() + 2 * shard_size},
                      shard_size,
                      {lanes[0], lanes[1], lanes[2]}};
    fill<<<(n + 255) / 256, 256, 0, streams[0].get()>>>(xbuf.data(), n, 1);
    streams[0].sync();

    // The pipeline. It begins on lane 0: that is the lane the reduce allocates
    // its partials on and forks the other lanes from.
    auto pipeline = ex::schedule(lanes[0]) | ex::let_value([&] {
                      // Inside this scope the environment's scheduler is lane 0.
                      return ex::read_env(cuda::mr::get_memory_resource) | ex::let_value([&](auto mr) {
                               return start(x, mr) //
                                    | transform(x, y, times2{}) //
                                    | reduce(y, result.data());
                             });
                    });
    // Everything in the pipeline that allocates -- the reduce's partials and
    // CUB's scratch -- takes its memory resource from this environment.
    using env_t = cuda::std::execution::env<
      cuda::std::execution::prop<cuda::mr::get_memory_resource_t, cuda::device_memory_pool_ref>>;
    const env_t env{{cuda::mr::get_memory_resource, mr}};

    const int expected = 2 * 1 * n; // 2 per element

    if (!as_graph)
    {
      // Eager: every `then` body above runs now, on the host, and enqueues onto
      // its lane. sync_wait returns when every lane is done.
      ex::sync_wait(std::move(pipeline), env);
    }
    else
    {
      // Captured: the same chain, enqueued into a capture that starts on lane 0.
      // The fork events bring the other lanes into the capture, the join events
      // become graph edges, and lane_capture joins every lane back at the end.
      // lane_capture: begin the capture on lane 0, run the pipeline, destroy its
      // operation state (so scoped frees are captured too), join every lane the
      // pipeline touched back into lane 0, end the capture.
      cudaGraph_t graph = ex::lane_capture(lanes[0], std::move(pipeline), env);
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
