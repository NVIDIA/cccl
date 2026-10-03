//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/functional>
#include <cuda/std/tuple>

#include <cuda/experimental/execution.cuh>
#include <cuda/experimental/stream.cuh>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

// nvcc cannot place __global__ kernels in the same anonymous namespace as the
// Catch2-generated test templates, so helpers live in a named namespace.
namespace lane_scheduler_test
{
__global__ void fill_k(int* p, int n, int v)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    p[i] = v;
  }
}
__global__ void sum2_k(const int* a, const int* b, int* out, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = a[i] + b[i];
  }
}
__global__ void spin_k(long long cycles)
{
  const long long t0 = clock64();
  while (clock64() - t0 < cycles)
  {
  }
}

struct times3
{
  __host__ __device__ int operator()(int x) const
  {
    return 3 * x;
  }
};

struct null_rcvr
{
  using receiver_concept = ex::receiver_t;
  void set_value() noexcept {}
  template <class E>
  void set_error(E&&) noexcept
  {}
  void set_stopped() noexcept {}
  ex::env<> get_env() const noexcept
  {
    return {};
  }
};

// Names a captured kernel node after the kernel it launches: "fill_a" / "fill_b"
// by the fill value, "sum" for sum2_k.
std::string node_name(cudaGraphNode_t node)
{
  cudaGraphNodeType type{};
  REQUIRE(cudaGraphNodeGetType(node, &type) == cudaSuccess);
  if (type != cudaGraphNodeTypeKernel)
  {
    return "other";
  }
  cudaKernelNodeParams p{};
  REQUIRE(cudaGraphKernelNodeGetParams(node, &p) == cudaSuccess);
  if (p.func == reinterpret_cast<const void*>(&sum2_k))
  {
    return "sum";
  }
  if (p.func == reinterpret_cast<const void*>(&fill_k))
  {
    return *static_cast<const int*>(p.kernelParams[2]) == 1 ? "fill_a" : "fill_b"; // 2 and 0 are lane b's fills
  }
  return "other";
}

// The dependency edges of a graph as "from->to" strings.
std::multiset<std::string> graph_edges(cudaGraph_t g)
{
  size_t n = 0;
  REQUIRE(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &n) == cudaSuccess);
  std::vector<cudaGraphNode_t> from(n), to(n);
  REQUIRE(cudaGraphGetEdges(g, from.data(), to.data(), nullptr, &n) == cudaSuccess);
  std::multiset<std::string> edges;
  for (size_t i = 0; i < n; ++i)
  {
    edges.insert(node_name(from[i]) + "->" + node_name(to[i]));
  }
  return edges;
}

// A memory resource that forwards to CUB's default (cudaMallocAsync on the stream)
// and counts the allocations, so a test can see the scratch allocations a CUB
// call makes when it is handed an environment.
struct counting_mr
{
  int* allocs;
  void* allocate_sync(size_t bytes, size_t align)
  {
    ++*allocs;
    return cub::detail::device_memory_resource{}.allocate(bytes, align);
  }
  void deallocate_sync(void* p, size_t bytes, size_t)
  {
    cub::detail::device_memory_resource{}.deallocate(p, bytes);
  }
  void* allocate(::cuda::stream_ref s, size_t bytes, size_t align)
  {
    ++*allocs;
    return cub::detail::device_memory_resource{}.allocate(s, bytes, align);
  }
  void deallocate(::cuda::stream_ref s, void* p, size_t bytes, size_t align)
  {
    cub::detail::device_memory_resource{}.deallocate(s, p, bytes, align);
  }
  friend constexpr void get_property(const counting_mr&, ::cuda::mr::device_accessible) noexcept {}
  bool operator==(const counting_mr& o) const
  {
    return allocs == o.allocs;
  }
  bool operator!=(const counting_mr& o) const
  {
    return !(*this == o);
  }
};

namespace lane
{
// cudax operation states are host/device, and nvcc's execution-space check
// rejects cuda::buffer's host-only destructor when it is reached from one. A
// buffer that travels as a sender value is therefore wrapped so that its
// (host-only) move and destruction happen inside functions exempted with the
// pragma cudax uses for its own tuples.
template <class T>
struct buffer
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE explicit buffer(::cuda::device_buffer<T>&& b)
      : buf_(::std::move(b))
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE buffer(buffer&& o) noexcept
      : buf_(::std::move(o.buf_))
  {}
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE ~buffer() {}
  T* data()
  {
    return buf_.data();
  }
  ::cuda::stream_ref stream() const
  {
    return buf_.stream();
  }

private:
  ::cuda::device_buffer<T> buf_;
};

// Allocation as a sender. Completes with a lane::buffer<T> (a cuda::device_buffer<T>)
// of n uninitialized elements allocated, stream-ordered, on the stream of the lane
// the sender runs on, through the memory resource of the environment; both are
// read back from the receiver's environment (get_scheduler, get_memory_resource
// -- forwarding queries, so they reach this sender at any depth). Hold it with
// let_value: the buffer lives in the let_value's operation state for the inner
// scope and is freed on the same stream when the scope ends. A buffer must
// therefore be allocated on the lane that is ordered after all of its readers
// -- the join target for shared data, the lane itself for scratch.
template <class T>
auto allocate(size_t n)
{
  return ex::when_all(ex::read_env(ex::get_scheduler), ex::read_env(::cuda::mr::get_memory_resource))
       | ex::then([n](auto sched, auto mr) {
           return buffer<T>{::cuda::device_buffer<T>{sched.query(::cuda::get_stream), mr, n, ::cuda::no_init}};
         });
}
} // namespace lane

// A per-test observer: counts the event joins issued by continues_on onto a lane.
struct join_counter
{
  int joins = 0;
  auto env()
  {
    auto observer = [this](cudaStream_t, cudaStream_t) {
      ++joins;
    };
    const auto prop = ::cuda::std::execution::prop{ex::get_lane_join_observer, observer};
    return ::cuda::std::execution::env{prop};
  }
};

struct fixture
{
  static constexpr int n    = 1 << 20;
  static constexpr int grid = (n + 255) / 256;
  cuda::stream sa{cuda::devices[0]};
  cuda::stream sb{cuda::devices[0]};
  ex::lane_scheduler la{sa.get()};
  ex::lane_scheduler lb{sb.get()};
  int *a{}, *b{}, *out{};
  fixture()
  {
    REQUIRE(cudaMalloc(&a, n * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMalloc(&b, n * sizeof(int)) == cudaSuccess);
    REQUIRE(cudaMalloc(&out, n * sizeof(int)) == cudaSuccess);
  }
  ~fixture()
  {
    cudaFree(a);
    cudaFree(b);
    cudaFree(out);
  }
  // Copies `out` back to the host and checks that every element equals `v`.
  bool all_equal(int v) const
  {
    std::vector<int> h(n);
    REQUIRE(cudaMemcpy(h.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    return ::std::all_of(h.begin(), h.end(), [v](int x) {
      return x == v;
    });
  }
};
} // namespace lane_scheduler_test

// C2H_TEST names its generated test by line number; keep the cases in an
// anonymous namespace (as the other execution tests do) so they cannot collide
// with same-line cases from other translation units.
namespace
{
using namespace lane_scheduler_test;

C2H_TEST("lane_scheduler: a single-lane chain issues no event", "[lane_scheduler]")
{
  fixture f;
  join_counter jc;
  auto chain =
    ex::schedule(f.la) //
    | ex::then([&] {
        fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 1);
      })
    | ex::then([&] {
        fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 2);
      });
  static_assert(ex::get_completion_behavior<decltype(chain)>() == ex::completion_behavior::synchronous);
  ex::sync_wait(std::move(chain) | ex::write_env(jc.env()));
  CHECK(jc.joins == 0);
}

C2H_TEST("lane_scheduler: when_all of two lanes + continues_on issues exactly one event", "[lane_scheduler]")
{
  fixture f;
  for (int target = 0; target < 2; ++target)
  {
    join_counter jc;
    auto lane_a         = ex::schedule(f.la) | ex::then([&] {
                    spin_k<<<1, 1, 0, f.sa.get()>>>(2000000);
                    fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 3);
                          });
    auto lane_b         = ex::schedule(f.lb) | ex::then([&] {
                    spin_k<<<1, 1, 0, f.sb.get()>>>(4000000);
                    fill_k<<<f.grid, 256, 0, f.sb.get()>>>(f.b, f.n, 4);
                          });
    auto& to            = target == 0 ? f.la : f.lb;
    cudaStream_t stream = target == 0 ? f.sa.get() : f.sb.get();
    auto joined         = ex::when_all(std::move(lane_a), std::move(lane_b)) //
                        | ex::continues_on(to) //
                        | ex::then([&, stream] {
                    sum2_k<<<f.grid, 256, 0, stream>>>(f.a, f.b, f.out, f.n);
                          });
    ex::sync_wait(std::move(joined), jc.env());
    CHECK(jc.joins == 1);
    // a = 3 and b = 4 were filled on different lanes; out = a + b is only 7
    // everywhere if the single event ordered both fills before the sum.
    CHECK(f.all_equal(7));
  }
}

C2H_TEST("lane_scheduler: a when_all under a lane forks from the when_all's start; fork and join become graph edges",
         "[lane_scheduler]")
{
  fixture f;
  cudaGraph_t g{};
  // The capture origin is lane a and the chain starts on it. The when_all is
  // under that lane, so it records one fork point on lane a before starting any
  // child; both forms of "begin on another lane" consume it: lane_b1 is a plain
  // schedule(lb), lane_b2 is schedule(la) | continues_on(lb). Neither depends on
  // lane a's own fill, which when_all starts first. The join back onto lane a is
  // continues_on. No hand-written cudaEventRecord/cudaStreamWaitEvent anywhere.
  REQUIRE(cudaStreamBeginCapture(f.sa.get(), cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  auto whole = ex::schedule(f.la) | ex::let_value([&] {
                 auto lane_a  = ex::schedule(f.la) | ex::then([&] {
                                 fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 1);
                                });
                 auto lane_b1 = ex::schedule(f.lb) | ex::then([&] {
                                  fill_k<<<f.grid, 256, 0, f.sb.get()>>>(f.b, f.n, 2);
                                });
                 auto lane_b2 = ex::schedule(f.la) | ex::continues_on(f.lb) | ex::then([&] {
                                  fill_k<<<f.grid, 256, 0, f.sb.get()>>>(f.out, f.n, 0);
                                });
                 return ex::when_all(std::move(lane_a), std::move(lane_b1), std::move(lane_b2))
                      | ex::continues_on(f.la) //
                      | ex::then([&] {
                          sum2_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.b, f.out, f.n);
                        });
               });
  // sync_wait would synchronize inside the capture; connect and start by hand.
  auto op = ex::connect(std::move(whole), null_rcvr{});
  ex::start(op);
  REQUIRE(cudaStreamEndCapture(f.sa.get(), &g) == cudaSuccess);
  // Expected: fill_a is a root, and so is the first lane-b fill (the fork point
  // on an empty capturing stream carries no node). The two lane-b fills share a
  // lane, so they serialize (fill_b -> fill_b): sharing a lane means ordering.
  // The sum depends on lane a and on lane b's tail. Nothing makes lane b wait
  // for lane a's fill, which when_all started first. Driver versions differ on
  // reporting transitive edges, so check the relation.
  size_t nodes = 0;
  REQUIRE(cudaGraphGetNodes(g, nullptr, &nodes) == cudaSuccess);
  CHECK(nodes == 4);
  const auto edges = graph_edges(g);
  CAPTURE(edges);
  CHECK(edges.count("fill_a->sum") == 1);
  CHECK(edges.count("fill_b->fill_b") == 1);
  CHECK(edges.count("fill_b->sum") >= 1);
  CHECK(edges.count("fill_a->fill_b") == 0); // the fork did not serialize lane b behind lane a
  for (const auto& e : edges)
  {
    CHECK((e == "fill_a->sum" || e == "fill_b->sum" || e == "fill_b->fill_b"));
  }
  cudaGraphExec_t ge{};
  REQUIRE(cudaGraphInstantiate(&ge, g, 0) == cudaSuccess);
  REQUIRE(cudaGraphLaunch(ge, f.sa.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(f.sa.get()) == cudaSuccess);
  // a = 1 on lane a, b = 2 on lane b, out = a + b after the join.
  CHECK(f.all_equal(3));
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
}

C2H_TEST("lane_scheduler: sync_wait waits for the lane's stream", "[lane_scheduler]")
{
  fixture f;
  REQUIRE(cudaMemset(f.a, 0, f.n * sizeof(int)) == cudaSuccess);
  auto chain = ex::schedule(f.la) | ex::then([&] {
                 spin_k<<<1, 1, 0, f.sa.get()>>>(200000000);
                 fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 9);
               });
  ex::sync_wait(std::move(chain));
  int h0 = -1;
  cuda::stream sc{cuda::devices[0]};
  REQUIRE(cudaMemcpyAsync(&h0, f.a, sizeof(int), cudaMemcpyDeviceToHost, sc.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(sc.get()) == cudaSuccess);
  CHECK(h0 == 9);
}
} // namespace

C2H_TEST("lane_scheduler: transform then reduce over lane-resident spans, allocations are senders", "[lane_scheduler]")
{
  // Two lanes, each owning one span (fixture a on lane a, fixture b on lane b).
  // Every allocation is a sender held by a let_value scope:
  //  - `partials` (one slot per lane) is allocated on lane a, the join target,
  //    so that its free on lane a's stream is ordered after the sum that reads it;
  //  - each lane's transform scratch is allocated on that lane and only used there.
  // Each lane runs CUB's Transform into its scratch, then CUB's Reduce into its
  // partial; CUB gets the stream from the scratch buffer and the memory resource
  // from the environment. The join onto lane a adds the partials.
  // The memory resource (a counting one) and the dependency observer are both
  // given once, as the environment of sync_wait; being forwarding queries they
  // reach every allocation and every cross-lane dependency in the chain: the
  // when_all under lane a forks lane b from its start (one event), and the
  // continues_on joins lane b back (one event). Nothing in the chain calls
  // cudaMalloc/cudaFree, and the counting resource sees every allocation: the
  // three buffers and one CUB Reduce scratch per lane.
  fixture f;
  join_counter jc;
  fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 1);
  fill_k<<<f.grid, 256, 0, f.sb.get()>>>(f.b, f.n, 2);

  int allocs  = 0;
  const int n = f.n;
  auto stage  = [=](auto& lane, const int* in, int* partial) {
    return ex::schedule(lane) | ex::let_value([=] {
             return lane::allocate<int>(n) | ex::let_value([=](lane::buffer<int>& scratch) {
                      return ex::read_env(::cuda::mr::get_memory_resource) | ex::then([=, &scratch](auto mr) {
                               auto env = cuda::std::execution::env{
                                 cuda::std::execution::prop{::cuda::get_stream, scratch.stream()},
                                 cuda::std::execution::prop{::cuda::mr::get_memory_resource, mr}};
                               REQUIRE(cub::DeviceTransform::Transform(
                                         cuda::std::make_tuple(in), scratch.data(), n, times3{}, env)
                                       == cudaSuccess);
                               REQUIRE(
                                 cub::DeviceReduce::Reduce(scratch.data(), partial, n, cuda::std::plus<>{}, 0, env)
                                 == cudaSuccess);
                             });
                    });
           });
  };
  auto whole = ex::schedule(f.la) | ex::let_value([&] {
                 return lane::allocate<int>(2) | ex::let_value([&](lane::buffer<int>& partials) {
                          return ex::when_all(stage(f.la, f.a, partials.data()), stage(f.lb, f.b, partials.data() + 1))
                               | ex::continues_on(f.la) //
                               | ex::then([&] {
                                   sum2_k<<<1, 1, 0, f.sa.get()>>>(partials.data(), partials.data() + 1, f.out, 1);
                                 });
                        });
               });
  ex::sync_wait(std::move(whole),
                cuda::std::execution::env{
                  jc.env(), cuda::std::execution::prop{::cuda::mr::get_memory_resource, counting_mr{&allocs}}});

  int result = 0;
  REQUIRE(cudaMemcpy(&result, f.out, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
  CHECK(result == 3 * 1 * f.n + 3 * 2 * f.n);
  CHECK(jc.joins == 2); // the fork point a -> b at the when_all, and the join b -> a
  CAPTURE(allocs);
  // partials + 2 scratch buffers + 1 CUB Reduce scratch per lane; Transform needs none.
  CHECK(allocs == 5);
}
