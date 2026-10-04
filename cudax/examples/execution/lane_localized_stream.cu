//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Localized STREAM, reduce and scan as senders, on lane schedulers over the
// device's locality domains.
//
// A device with locality domains (two dies behind one device) serves memory
// fastest from the die it is attached to. A *localized* algorithm keeps each
// shard's kernels on the die that holds the shard's bytes: the shard's lane is a
// stream on the domain's green context (`cuda::stream{domain}`), and its memory
// comes from the domain's pool (`cuda::__device_default_memory_pool(domain)`).
// The lane scheduler never knows: a lane is a stream, whatever context it
// belongs to.
//
// The difference from lane_sharded_pipeline.cu is where the memory resource comes
// from. There, one resource is read from the sender environment at the root and
// serves every shard. A localized array has one pool *per shard*, so here a
// shard's context is its *place*, the lane and the pool, taken from the sharded
// view when the pipeline starts; no verb reads the environment. (A side effect:
// nothing below the root is environment-dependent, which keeps the compile time
// of the verbs flat.)
//
// The verbs are the STREAM kernels, copy, scale, add and triad, as elementwise
// `transform`s (one CUB call per shard on its own lane, no event anywhere), a
// `reduce` (one fork, one join, partials as scoped scratch on the join lane) and
// an in-place `inclusive_scan` (collapse to one lane for the carries, re-fork
// through a lane_split). The example runs them on two arms, `whole` (one lane,
// one interleaved allocation) and `localized` (one lane per domain), checks the
// results and reports the delivered bandwidth of each. On a device without
// locality domains both arms are the same kernels on the same memory.
//
//   lane_localized_stream [--pow=N] [--graph]
//
// --pow=N     2^N elements per array (default 24)
// --graph     also capture `add | triad | reduce` through lane_capture and
//             replay it
//
// Compile time: a chain of verbs nests one sender adaptor per verb, and nvcc's
// device front end grows super-linearly with the depth even without any
// environment-dependent sender (measured on this file with -arch=native: the
// captured chain at 3 verbs, 53 s; at 5 verbs, `copy | scale | add | triad |
// reduce`, 13 min). The captured chain is kept at 3 verbs for that reason.

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/argument>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_resource>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/execution.cuh>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

namespace ex = cuda::experimental::execution;

void check(cudaError_t st, const char* what)
{
  if (st != cudaSuccess)
  {
    throw cuda::cuda_error(st, what);
  }
}

// ----------------------------------------------------------------------------
// A place: where a shard's kernels run (its lane) and where its bytes live (its
// pool). Default constructible, so that arrays of places can be filled in.
struct place
{
  ex::lane_scheduler lane;
  cudaMemPool_t pool{};

  cuda::device_memory_pool_ref mr() const
  {
    return cuda::device_memory_pool_ref{pool};
  }
  cudaStream_t stream() const
  {
    return lane.stream();
  }
  // The environment a CUB call needs on this place: the stream, and the pool
  // for CUB's own scratch.
  auto cub_env() const
  {
    using stream_prop = cuda::std::execution::prop<cuda::get_stream_t, cuda::stream_ref>;
    using mr_prop     = cuda::std::execution::prop<cuda::mr::get_memory_resource_t, cuda::device_memory_pool_ref>;
    return cuda::std::execution::env<stream_prop, mr_prop>{
      stream_prop{cuda::get_stream, lane.query(cuda::get_stream)}, mr_prop{cuda::mr::get_memory_resource, mr()}};
  }
};

// A sharded array: N shards, each a span on a place.
template <class T, size_t N>
struct sharded_view
{
  T* data[N];
  size_t shard_size;
  place at[N];
};

// A bundle of per-shard senders, what verbs take and return.
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

// A device_buffer that may travel through host/device operation states (see
// lane_sharded_pipeline.cu).
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

// Scoped allocation on a known place, as a sender: the body runs at start(), so
// under capture the allocation is captured with the work that uses it.
template <class T>
auto allocate_at(place p, size_t n)
{
  return ex::just() | ex::then([=] {
           return scoped_buffer<T>{cuda::device_buffer<T>{cuda::stream_ref{p.stream()}, p.mr(), n, cuda::no_init}};
         });
}

// ----------------------------------------------------------------------------
// Verbs: closures over bundles, composable with `|`.
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
template <class F, class G>
auto operator|(verb<F> f, verb<G> g)
{
  return verb{[f, g](auto b) {
    return g.fn(f.fn(std::move(b)));
  }};
}

// start(view): begin on every shard's lane, completing with the shard's place.
template <class T, size_t N, size_t... I>
auto start(const sharded_view<T, N>& v, cuda::std::index_sequence<I...>)
{
  auto on = [](place p) {
    return ex::schedule(p.lane) | ex::then([=] {
             return p;
           });
  };
  return bundle{cuda::std::make_tuple(on(v.at[I])...)};
}
template <class T, size_t N>
auto start(const sharded_view<T, N>& v)
{
  return start(v, cuda::std::make_index_sequence<N>{});
}

// finish(bundle): one sender over the bundle. Directly under sync_wait, with no
// lane in the environment, this forks nothing; sync_wait waits on every lane.
template <class... S>
auto finish(bundle<S...> b)
{
  return cuda::std::apply(
    [](auto&&... s) {
      return ex::when_all(std::move(s)...);
    },
    std::move(b.shards));
}

// Elementwise verbs: out[k] = op(in...[k]) on shard k's place. No lane meets another.
template <class T, size_t N, class Op>
auto transform(const sharded_view<T, N>& in, const sharded_view<T, N>& out, Op op)
{
  return verb{[=](auto b) {
    return map(
      std::move(b.shards),
      [=](auto shard, size_t k) {
        return std::move(shard) | ex::then([=](place p) {
                 check(cub::DeviceTransform::Transform(
                         cuda::std::make_tuple(in.data[k]), out.data[k], in.shard_size, op, p.cub_env()),
                       "DeviceTransform::Transform");
                 return p;
               });
      },
      cuda::std::make_index_sequence<N>{});
  }};
}
template <class T, size_t N, class Op>
auto transform(const sharded_view<T, N>& in1, const sharded_view<T, N>& in2, const sharded_view<T, N>& out, Op op)
{
  return verb{[=](auto b) {
    return map(
      std::move(b.shards),
      [=](auto shard, size_t k) {
        return std::move(shard) | ex::then([=](place p) {
                 check(cub::DeviceTransform::Transform(
                         cuda::std::make_tuple(in1.data[k], in2.data[k]), out.data[k], in1.shard_size, op, p.cub_env()),
                       "DeviceTransform::Transform");
                 return p;
               });
      },
      cuda::std::make_index_sequence<N>{});
  }};
}

// STREAM: copy c = a, scale b = s c, add c = a + b, triad a = b + s c.
template <class T>
struct copy_op
{
  __host__ __device__ T operator()(T a) const
  {
    return a;
  }
};
template <class T>
struct scale_op
{
  T s;
  __host__ __device__ T operator()(T c) const
  {
    return s * c;
  }
};
template <class T>
struct add_op
{
  __host__ __device__ T operator()(T a, T b) const
  {
    return a + b;
  }
};
template <class T>
struct triad_op
{
  T s;
  __host__ __device__ T operator()(T b, T c) const
  {
    return b + s * c;
  }
};
template <class T, size_t N>
auto stream_copy(const sharded_view<T, N>& a, const sharded_view<T, N>& c)
{
  return transform(a, c, copy_op<T>{});
}
template <class T, size_t N>
auto stream_scale(const sharded_view<T, N>& c, const sharded_view<T, N>& b, T s)
{
  return transform(c, b, scale_op<T>{s});
}
template <class T, size_t N>
auto stream_add(const sharded_view<T, N>& a, const sharded_view<T, N>& b, const sharded_view<T, N>& c)
{
  return transform(a, b, c, add_op<T>{});
}
template <class T, size_t N>
auto stream_triad(const sharded_view<T, N>& b, const sharded_view<T, N>& c, const sharded_view<T, N>& a, T s)
{
  return transform(b, c, a, triad_op<T>{s});
}

// reduce(in, result): each shard reduces into its own partial on its own place;
// the lanes meet once on shard 0's place, where the partials live (a scoped
// allocation, freed after the sum that reads them) and are added into `result`.
template <class T>
__global__ void sum_partials(const T* partials, int n, T* result)
{
  T acc{};
  for (int i = 0; i < n; ++i)
  {
    acc += partials[i];
  }
  *result = acc;
}
template <class Bundle, class T, size_t N, size_t... I>
auto reduce_impl(Bundle b, const sharded_view<T, N>& in, T* result, cuda::std::index_sequence<I...>)
{
  const place home = in.at[0];
  return allocate_at<T>(home, N) //
       | ex::let_value([b = std::move(b), in, result, home](scoped_buffer<T>& partials) mutable {
           auto shard_reduce = [=, p = partials.data()](auto shard, size_t k) {
             return std::move(shard) | ex::then([=](place pl) {
                      check(cub::DeviceReduce::Reduce(
                              in.data[k], p + k, in.shard_size, cuda::std::plus<>{}, T{}, pl.cub_env()),
                            "DeviceReduce::Reduce");
                    });
           };
           return ex::when_all(shard_reduce(cuda::std::get<I>(std::move(b.shards)), I)...) // the fork
                | ex::continues_on(home.lane) // the join
                | ex::then([=, p = partials.data()] {
                    sum_partials<T><<<1, 1, 0, home.stream()>>>(p, static_cast<int>(N), result);
                  });
         });
}
template <class T, size_t N>
auto reduce(const sharded_view<T, N>& in, T* result)
{
  return verb{[=](auto b) {
    return reduce_impl(std::move(b), in, result, cuda::std::make_index_sequence<N>{});
  }};
}

// inclusive_scan(data), in place across the global index space: per-shard
// totals (through lane_split, so that each total's buffer outlives its reader),
// collapse onto shard 0's place for the carries, re-fork through a lane_split of
// the prefix, one InclusiveScanInit per shard seeded with its carry read on the
// device. N-1 events to collapse and N-1 to re-fork; the result is a bundle again.
template <class T, size_t N>
struct total_ptrs
{
  const T* p[N];
};
template <class T, size_t N>
__global__ void prefix_kernel(total_ptrs<T, N> totals, T* carries)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    T acc{};
    for (size_t k = 0; k < N; ++k)
    {
      carries[k] = acc;
      acc += *totals.p[k];
    }
  }
}
template <class T>
struct shard_total
{
  const T* total;
  place at;
};
template <class T, size_t N>
struct scan_carries
{
  const T* carries;
  place at[N];
};
template <class Bundle, class T, size_t N, size_t... I>
auto inclusive_scan_impl(Bundle b, const sharded_view<T, N>& data, cuda::std::index_sequence<I...>)
{
  auto total = [=](auto shard, size_t k) {
    return std::move(shard) | ex::let_value([=](place p) {
             return allocate_at<T>(p, 1) | ex::let_value([=](scoped_buffer<T>& t) {
                      check(cub::DeviceReduce::Reduce(
                              data.data[k], t.data(), data.shard_size, cuda::std::plus<>{}, T{}, p.cub_env()),
                            "DeviceReduce::Reduce");
                      return ex::just(shard_total<T>{t.data(), p});
                    });
           });
  };
  const place home = data.at[0];
  auto totals      = cuda::std::make_tuple(ex::lane_split(total(cuda::std::get<I>(std::move(b.shards)), I))...);
  auto prefix      = ex::lane_split(
    ex::when_all(cuda::std::get<I>(totals)...) //
    | ex::continues_on(home.lane) //
    | ex::let_value([=](auto... t) {
        return allocate_at<T>(home, N) | ex::let_value([=](scoped_buffer<T>& carries) {
                 prefix_kernel<T, N><<<1, 32, 0, home.stream()>>>(total_ptrs<T, N>{{t.total...}}, carries.data());
                 return ex::just(scan_carries<T, N>{carries.data(), {t.at...}});
               });
      }));
  auto stage = [=](auto k) {
    constexpr size_t K = decltype(k)::value;
    return prefix | ex::continues_on(data.at[K].lane) | ex::then([=](auto sc) {
             place p = sc.at[K];
             check(cub::DeviceScan::InclusiveScanInit(
                     data.data[K],
                     data.data[K],
                     cuda::std::plus<>{},
                     cuda::args::deferred(sc.carries + K),
                     data.shard_size,
                     p.cub_env()),
                   "DeviceScan::InclusiveScanInit");
             return p;
           });
  };
  return bundle{cuda::std::make_tuple(stage(cuda::std::integral_constant<size_t, I>{})...)};
}
template <class T, size_t N>
auto inclusive_scan(const sharded_view<T, N>& data)
{
  return verb{[=](auto b) {
    return inclusive_scan_impl(std::move(b), data, cuda::std::make_index_sequence<N>{});
  }};
}

// ----------------------------------------------------------------------------
// The driver.
template <class T>
__global__ void fill(T* p, size_t n, T v)
{
  const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    p[i] = v;
  }
}

template <class Fn>
double median_ms(Fn&& fn, int warmup = 2, int reps = 7)
{
  for (int i = 0; i < warmup; ++i)
  {
    fn();
  }
  std::vector<double> t;
  for (int i = 0; i < reps; ++i)
  {
    const auto t0 = std::chrono::steady_clock::now();
    fn();
    const auto t1 = std::chrono::steady_clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

// A sharded array allocated on the given places, one shard per place, filled.
template <class T, size_t N>
struct sharded_array
{
  std::vector<cuda::device_buffer<T>> bufs;
  sharded_view<T, N> view;

  sharded_array(const place (&at)[N], size_t shard_size, T init)
  {
    view.shard_size = shard_size;
    for (size_t k = 0; k < N; ++k)
    {
      bufs.emplace_back(cuda::stream_ref{at[k].stream()}, at[k].mr(), shard_size, cuda::no_init);
      view.data[k] = bufs.back().data();
      view.at[k]   = at[k];
      refill(k, init);
    }
  }
  void refill(size_t k, T v)
  {
    const unsigned grid = static_cast<unsigned>((view.shard_size + 255) / 256);
    fill<T><<<grid, 256, 0, view.at[k].stream()>>>(view.data[k], view.shard_size, v);
  }
};

// Runs STREAM, reduce and scan over the N places, checks, reports. Returns false
// on a wrong result.
template <class T, size_t N>
bool run(const char* arm, const place (&at)[N], size_t n, bool capture)
{
  const size_t shard_size = n / N;
  const T s{3};
  sharded_array<T, N> a{at, shard_size, T{1}};
  sharded_array<T, N> b{at, shard_size, T{2}};
  sharded_array<T, N> c{at, shard_size, T{0}};
  sharded_array<int, N> scanned{at, shard_size, 1}; // int: the scan is checked exactly
  cuda::device_buffer<T> result{cuda::stream_ref{at[0].stream()}, at[0].mr(), 1, cuda::no_init};
  for (const place& p : at)
  {
    check(cudaStreamSynchronize(p.stream()), "cudaStreamSynchronize");
  }
  const auto& av = a.view;
  const auto& bv = b.view;
  const auto& cv = c.view;
  const auto& sv = scanned.view;

  // Each STREAM kernel alone, as its own pipeline. Elementwise: one chain per
  // lane, no event. From a = 1, b = 2, c = 0 the four kernels are idempotent
  // once each has run once: c = 1, b = 3, c = 4, a = 15.
  auto time_bundle = [&](auto make) {
    return median_ms([&] {
      ex::sync_wait(finish(make()));
    });
  };
  const double ms_copy  = time_bundle([&] {
    return start(av) | stream_copy(av, cv);
  });
  const double ms_scale = time_bundle([&] {
    return start(cv) | stream_scale(cv, bv, s);
  });
  const double ms_add   = time_bundle([&] {
    return start(av) | stream_add(av, bv, cv);
  });
  const double ms_triad = time_bundle([&] {
    return start(bv) | stream_triad(bv, cv, av, s);
  });
  // The reduce joins on shard 0's lane; the pipeline begins there so that the
  // fork is from it.
  const double ms_reduce = median_ms([&] {
    ex::sync_wait(ex::schedule(at[0].lane) | ex::let_value([&] {
                    return start(av) | reduce(av, result.data());
                  }));
  });
  const double ms_scan   = median_ms([&] {
    ex::sync_wait(ex::schedule(at[0].lane) | ex::let_value([&] {
                    return finish(start(sv) | inclusive_scan(sv));
                  }));
  });

  T r{};
  check(cudaMemcpy(&r, result.data(), sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
  bool ok = r == T{15} * static_cast<T>(n);
  // The scan is in place and ran repeatedly: refill, scan once, check the last
  // element and the first element of the last shard (the carry).
  for (size_t k = 0; k < N; ++k)
  {
    scanned.refill(k, 1);
  }
  ex::sync_wait(ex::schedule(at[0].lane) | ex::let_value([&] {
                  return finish(start(sv) | inclusive_scan(sv));
                }));
  int last = 0, carried = 0;
  check(cudaMemcpy(&last, sv.data[N - 1] + shard_size - 1, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  check(cudaMemcpy(&carried, sv.data[N - 1], sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
  ok = ok && last == static_cast<int>(n) && carried == static_cast<int>((N - 1) * shard_size + 1);

  const double bytes2 = 2.0 * sizeof(T) * n; // copy, scale: one read, one write
  const double bytes3 = 3.0 * sizeof(T) * n; // add, triad: two reads, one write
  auto gbps           = [](double bytes, double ms) {
    return bytes / (ms * 1e-3) / 1e9;
  };
  std::printf(
    "%-9s %zu lane(s)  copy %6.0f  scale %6.0f  add %6.0f  triad %6.0f  reduce %6.0f  scan %6.0f GB/s  "
    "[%s]\n",
    arm,
    N,
    gbps(bytes2, ms_copy),
    gbps(bytes2, ms_scale),
    gbps(bytes3, ms_add),
    gbps(bytes3, ms_triad),
    gbps(1.0 * sizeof(T) * n, ms_reduce),
    gbps(2.0 * sizeof(int) * n, ms_scan),
    ok ? "ok" : "WRONG");

  if (capture)
  {
    // Two STREAM kernels and the reduce as one algorithm, captured once through
    // lane_capture and replayed. The elementwise verbs chain per lane; only the
    // reduce joins. Replays change a, b, c; the check refills a = 1, b = 2,
    // c = 0 and runs once: c = 3, a = 2 + 9 = 11.
    auto algo         = stream_add(av, bv, cv) | stream_triad(bv, cv, av, s) | reduce(av, result.data());
    auto pipeline     = ex::schedule(at[0].lane) | ex::let_value([&] {
                      return start(av) | algo;
                        });
    cudaGraph_t graph = ex::lane_capture(at[0].lane, std::move(pipeline));
    size_t nodes = 0, edges = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes), "cudaGraphGetNodes");
    check(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges), "cudaGraphGetEdges");
    cudaGraphExec_t exec{};
    check(cudaGraphInstantiate(&exec, graph, 0), "cudaGraphInstantiate");
    const double ms_graph = median_ms([&] {
      check(cudaGraphLaunch(exec, at[0].stream()), "cudaGraphLaunch");
      check(cudaStreamSynchronize(at[0].stream()), "cudaStreamSynchronize");
    });
    for (size_t k = 0; k < N; ++k)
    {
      a.refill(k, T{1});
      b.refill(k, T{2});
      c.refill(k, T{0});
      check(cudaStreamSynchronize(at[k].stream()), "cudaStreamSynchronize");
    }
    check(cudaGraphLaunch(exec, at[0].stream()), "cudaGraphLaunch");
    check(cudaStreamSynchronize(at[0].stream()), "cudaStreamSynchronize");
    check(cudaMemcpy(&r, result.data(), sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    const bool ok_graph = r == T{11} * static_cast<T>(n);
    ok                  = ok && ok_graph;
    std::printf(
      "%-9s %zu lane(s)  add|triad|reduce as one graph: %zu nodes, %zu edges, replay %6.0f GB/s "
      "[%s]\n",
      arm,
      N,
      nodes,
      edges,
      gbps(2 * bytes3 + sizeof(T) * n, ms_graph),
      ok_graph ? "ok" : "WRONG");
    check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
  }
  return ok;
}

int main(int argc, char** argv)
{
  try
  {
    int pow      = 24;
    bool capture = false;
    for (int i = 1; i < argc; ++i)
    {
      if (!std::strcmp(argv[i], "--graph"))
      {
        capture = true;
      }
      else if (!std::strncmp(argv[i], "--pow=", 6))
      {
        pow = std::atoi(argv[i] + 6);
      }
    }
    const size_t n = size_t{1} << pow;
    using T        = double;

    cuda::device_ref dev{0};
    const auto domains = dev.__locality_domains();
    std::printf("device 0: %zu locality domain(s), %zu doubles per array\n", domains.size(), n);

    // whole: one lane on a device stream, memory from the device's pool, which
    // interleaves across the dies.
    cuda::stream whole_stream{dev};
    const place whole[1] = {place{ex::lane_scheduler{whole_stream}, cuda::device_default_memory_pool(dev).get()}};
    bool ok              = run<T, 1>("whole", whole, n, capture);

    // localized: one lane per locality domain, a stream on the domain's green
    // context, memory from the domain's pool. Two domains are the shape this
    // example is about; with one, the arm is `whole` again with another stream.
    if (domains.size() == 2)
    {
      cuda::stream domain_streams[2] = {cuda::stream{domains[0]}, cuda::stream{domains[1]}};
      const place localized[2]       = {
        place{ex::lane_scheduler{domain_streams[0]}, cuda::__device_default_memory_pool(domains[0]).get()},
        place{ex::lane_scheduler{domain_streams[1]}, cuda::__device_default_memory_pool(domains[1]).get()}};
      ok = run<T, 2>("localized", localized, n, capture) && ok;
    }
    else
    {
      cuda::stream other{dev};
      const place two[2] = {whole[0], place{ex::lane_scheduler{other}, whole[0].pool}};
      ok                 = run<T, 2>("two-lane", two, n, capture) && ok;
    }
    std::printf("%s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
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
