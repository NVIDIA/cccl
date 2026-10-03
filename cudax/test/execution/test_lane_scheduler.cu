//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/experimental/execution.cuh>
#include <cuda/experimental/stream.cuh>

#include <vector>

#include "testing.cuh" // IWYU pragma: keep

namespace ex = cuda::experimental::execution;

namespace
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
  int count_not(int v) const
  {
    std::vector<int> h(n);
    REQUIRE(cudaMemcpy(h.data(), out, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);
    int bad = 0;
    for (int x : h)
    {
      bad += (x != v);
    }
    return bad;
  }
};
} // namespace

C2H_TEST("lane_scheduler: a single-lane chain issues no event", "[lane_scheduler]")
{
  fixture f;
  ex::__lane::joins_issued() = 0;
  auto chain = ex::schedule(f.la) //
             | ex::then([&] {
                 fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 1);
               })
             | ex::then([&] {
                 fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 2);
               });
  static_assert(ex::get_completion_behavior<decltype(chain)>() == ex::completion_behavior::synchronous);
  ex::sync_wait(std::move(chain));
  CHECK(ex::__lane::joins_issued() == 0);
}

C2H_TEST("lane_scheduler: when_all of two lanes + continues_on issues exactly one event", "[lane_scheduler]")
{
  fixture f;
  for (int target = 0; target < 2; ++target)
  {
    ex::__lane::joins_issued() = 0;
    auto lane_a = ex::schedule(f.la) | ex::then([&] {
                    spin_k<<<1, 1, 0, f.sa.get()>>>(2000000);
                    fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 3);
                  });
    auto lane_b = ex::schedule(f.lb) | ex::then([&] {
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
    ex::sync_wait(std::move(joined));
    CHECK(ex::__lane::joins_issued() == 1);
    CHECK(f.count_not(7) == 0);
  }
}

C2H_TEST("lane_scheduler: the lazy join becomes a graph edge under stream capture", "[lane_scheduler]")
{
  fixture f;
  cudaGraph_t g{};
  cudaEvent_t fork{};
  REQUIRE(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming) == cudaSuccess);
  REQUIRE(cudaStreamBeginCapture(f.sa.get(), cudaStreamCaptureModeThreadLocal) == cudaSuccess);
  REQUIRE(cudaEventRecord(fork, f.sa.get()) == cudaSuccess);
  REQUIRE(cudaStreamWaitEvent(f.sb.get(), fork, 0) == cudaSuccess);
  auto lane_a = ex::schedule(f.la) | ex::then([&] {
                  fill_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.n, 1);
                });
  auto lane_b = ex::schedule(f.lb) | ex::then([&] {
                  fill_k<<<f.grid, 256, 0, f.sb.get()>>>(f.b, f.n, 2);
                });
  auto joined = ex::when_all(std::move(lane_a), std::move(lane_b)) | ex::continues_on(f.la) | ex::then([&] {
                  sum2_k<<<f.grid, 256, 0, f.sa.get()>>>(f.a, f.b, f.out, f.n);
                });
  // sync_wait would synchronize inside the capture; connect and start by hand.
  auto op = ex::connect(std::move(joined), null_rcvr{});
  ex::start(op);
  REQUIRE(cudaStreamEndCapture(f.sa.get(), &g) == cudaSuccess);
  size_t nodes = 0, edges = 0;
  REQUIRE(cudaGraphGetNodes(g, nullptr, &nodes) == cudaSuccess);
  REQUIRE(cudaGraphGetEdges(g, nullptr, nullptr, nullptr, &edges) == cudaSuccess);
  CHECK(nodes == 3);
  CHECK(edges == 2);
  cudaGraphExec_t ge{};
  REQUIRE(cudaGraphInstantiate(&ge, g, 0) == cudaSuccess);
  REQUIRE(cudaGraphLaunch(ge, f.sa.get()) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(f.sa.get()) == cudaSuccess);
  CHECK(f.count_not(3) == 0);
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaEventDestroy(fork);
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
