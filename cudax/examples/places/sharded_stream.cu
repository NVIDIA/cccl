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
 * @brief The STREAM benchmark (copy, scale, add, triad) over the locality
 *        domains of one device, with the sharded tier, against the same
 *        Thrust kernels on the whole device.
 *
 * Mapping-tier example:
 *  1. `place_group::by_locality_domains()` — one execution place per locality
 *     domain (a single whole-device place where domains are unsupported);
 *  2. `sharded_array<T>::allocate(group, n)` — each array is split into one
 *     shard per place, allocated in that place's memory;
 *  3. `sharded::zip_transform` runs each STREAM kernel per shard on the shard's
 *     own stream (environments derived from the output array), so every place
 *     streams its own memory. The whole-device
 *     arm runs the identical Thrust kernel on one interleaved allocation.
 *
 * Both arms are checked against the analytic STREAM values and reported as
 * GB/s (bytes moved / wall-clock of the call, median of several repetitions).
 * On dies with locality domains, the sharded arm keeps the inter-die fabric
 * idle and typically streams faster; on a single-domain device both arms
 * are the same kernel and should read about the same.
 *
 * Usage: sharded_stream [log2_elements=27] [reps=7]
 */

#include <cuda/experimental/sharded.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
using T = double; // canonical STREAM element; float overstates the locality ratio (narrow elements pay the stitch latency)
constexpr T scalar = T{3};

struct copy_op
{
  __host__ __device__ T operator()(T a) const
  {
    return a;
  }
};
struct scale_op
{
  __host__ __device__ T operator()(T c) const
  {
    return scalar * c;
  }
};
struct add_op
{
  __host__ __device__ T operator()(T a, T b) const
  {
    return a + b;
  }
};
struct triad_op
{
  __host__ __device__ T operator()(T b, T c) const
  {
    return b + scalar * c;
  }
};

// Host wall-clock around the call plus a device sync: what a caller sees.
template <typename F>
double median_ms(int reps, F&& f)
{
  for (int i = 0; i < 2; i++)
  {
    f();
    cuda_safe_call(cudaDeviceSynchronize());
  }
  std::vector<double> t;
  for (int i = 0; i < reps; i++)
  {
    cuda_safe_call(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    f();
    cuda_safe_call(cudaDeviceSynchronize());
    t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

bool all_equal(const std::vector<T>& h, T v, const char* what)
{
  for (std::size_t i = 0; i < h.size(); i++)
  {
    if (h[i] != v)
    {
      printf("FAILED %s: element %zu = %g, expected %g\n", what, i, double(h[i]), double(v));
      return false;
    }
  }
  return true;
}

struct row
{
  const char* name;
  double bytes_per_elem;
  double whole_ms;
  double sharded_ms;
};

void print(const row& r, std::size_t n)
{
  const double bytes = r.bytes_per_elem * double(n);
  printf("%-6s whole %8.3f ms (%6.0f GB/s)   sharded %8.3f ms (%6.0f GB/s)   sharded/whole speedup %.2fx\n",
         r.name,
         r.whole_ms,
         bytes / r.whole_ms / 1e6,
         r.sharded_ms,
         bytes / r.sharded_ms / 1e6,
         r.whole_ms / r.sharded_ms);
}
} // namespace

int main(int argc, char** argv)
{
  const int log2n       = argc > 1 ? atoi(argv[1]) : 27;
  const int reps        = argc > 2 ? atoi(argv[2]) : 7;
  const std::size_t n   = std::size_t{1} << log2n;
  const double bytes_rw = 2.0 * sizeof(T); // copy, scale: one read + one write per element
  const double bytes_rrw = 3.0 * sizeof(T); // add, triad: two reads + one write

  auto group = place_group{make_locality_domain_grid()};
  printf("STREAM, %zu doubles per array (%.1f GiB), %d reps, place_group with %zu place(s)\n",
         n,
         double(n) * sizeof(T) / (1u << 30),
         reps,
         group.size());

  bool ok = true;
  std::vector<T> host(n);

  // ---- whole-device arm: one interleaved allocation per array, one stream ----
  row rows[4] = {{"copy", bytes_rw, 0, 0}, {"scale", bytes_rw, 0, 0}, {"add", bytes_rrw, 0, 0}, {"triad", bytes_rrw, 0, 0}};
  {
    cudaStream_t s = nullptr;
    cuda_safe_call(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    T *a = nullptr, *b = nullptr, *c = nullptr;
    cuda_safe_call(cudaMalloc(&a, n * sizeof(T)));
    cuda_safe_call(cudaMalloc(&b, n * sizeof(T)));
    cuda_safe_call(cudaMalloc(&c, n * sizeof(T)));
    auto pol = thrust::cuda::par_nosync.on(s);
    thrust::fill(pol, a, a + n, T{1});
    thrust::fill(pol, b, b + n, T{2});
    thrust::fill(pol, c, c + n, T{0});

    // one checked pass: c = a = 1; b = 3c = 3; c = a + b = 4; a = b + 3c = 15
    thrust::transform(pol, a, a + n, c, copy_op{});
    thrust::transform(pol, c, c + n, b, scale_op{});
    thrust::transform(pol, a, a + n, b, c, add_op{});
    thrust::transform(pol, b, b + n, c, a, triad_op{});
    cuda_safe_call(cudaStreamSynchronize(s)); // non-blocking stream: the legacy-stream memcpy below does not wait for it
    cuda_safe_call(cudaMemcpy(host.data(), a, n * sizeof(T), cudaMemcpyDeviceToHost));
    ok = all_equal(host, T{15}, "whole-device triad") && ok;

    rows[0].whole_ms = median_ms(reps, [&] {
      thrust::transform(pol, a, a + n, c, copy_op{});
    });
    rows[1].whole_ms = median_ms(reps, [&] {
      thrust::transform(pol, c, c + n, b, scale_op{});
    });
    rows[2].whole_ms = median_ms(reps, [&] {
      thrust::transform(pol, a, a + n, b, c, add_op{});
    });
    rows[3].whole_ms = median_ms(reps, [&] {
      thrust::transform(pol, b, b + n, c, a, triad_op{});
    });

    cuda_safe_call(cudaFree(a));
    cuda_safe_call(cudaFree(b));
    cuda_safe_call(cudaFree(c));
    cuda_safe_call(cudaStreamDestroy(s));
  }

  // ---- sharded arm: one shard per locality domain, each on its own memory and stream ----
  {
    auto a = sharded_array<T>::allocate(group, n);
    auto b = sharded_array<T>::allocate(group, n);
    auto c = sharded_array<T>::allocate(group, n);
    fill(a, T{1});
    fill(b, T{2});
    fill(c, T{0});

    zip_transform(c, copy_op{}, a);
    zip_transform(b, scale_op{}, c);
    zip_transform(c, add_op{}, a, b);
    zip_transform(a, triad_op{}, b, c);
    a.copy_to_host(host.data());
    ok = all_equal(host, T{15}, "sharded triad") && ok;

    // Synchronous convenience form (barrier on the shard streams inside the call);
    // the harness's device sync makes both arms' timing identical in shape.
    rows[0].sharded_ms = median_ms(reps, [&] {
      zip_transform(c, copy_op{}, a);
    });
    rows[1].sharded_ms = median_ms(reps, [&] {
      zip_transform(b, scale_op{}, c);
    });
    rows[2].sharded_ms = median_ms(reps, [&] {
      zip_transform(c, add_op{}, a, b);
    });
    rows[3].sharded_ms = median_ms(reps, [&] {
      zip_transform(a, triad_op{}, b, c);
    });
  }

  for (const auto& r : rows)
  {
    print(r, n);
  }

  printf(ok ? "PASSED\n" : "FAILED\n");
  return ok ? 0 : 1;
}
