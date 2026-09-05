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
 * @brief Monte Carlo pi over sharded arrays: three tiers composing through
 *        one binding.
 *
 * The pipeline is vendor tier -> generic tier -> native tier, with nothing
 * connecting the stages but the sharded structures themselves:
 *
 *   generate_uniform(x/y)   — cuRAND tier (random.cuh, opt-in)
 *   zip_transform(inside)   — in-circle test per sample
 *   reduce(inside)          — hit count, then pi = 4 * hits / n
 *
 * Because generation is sharding-invariant (bitwise: sample k is a pure
 * function of (seed, k)), the estimate is REPRODUCIBLE ACROSS SHARDINGS:
 * this example computes pi twice with different shard boundaries and checks
 * the hit counts match exactly. (The sum of 0/1 float flags is integer-exact,
 * so even the reduction is grouping-independent.)
 *
 * Both spellings are shown, because pi is exactly the workload where a
 * reviewer should ask "why materialize at all?":
 *
 *  1. MATERIALIZED (above): three passes over memory. The right spelling
 *     when the random data must persist and be re-read (initialization,
 *     dropout masks, solver states) — placement at generation time then
 *     pays on every downstream read.
 *  2. FUSED: one kernel per shard, zero bytes of samples materialized —
 *     generation (device-API positional mapping), the in-circle test and
 *     the reduction in a single pass over the INDEX SPACE. Reshard-invariant
 *     by the same argument: sample k is a pure function of (seed, k).
 *     Today this is spelled per shard (CUB and kernels are iterator-native);
 *     making it generic over foreign structures is the lazy-input
 *     iterator-descriptor relaxation recorded in concepts.cuh's
 *     v1-simplifications note.
 *
 * The two estimates differ bitwise from each other (host-API float sequence
 * vs positional device mapping — documented in random.cuh), but EACH is
 * bitwise reproducible across shardings, which this example checks for both.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdio>

#include <cuda/experimental/__sharded/random.cuh> // opt-in vendor tier

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct in_unit_circle
{
  __host__ __device__ float operator()(float x, float y) const
  {
    return (x * x + y * y <= 1.0f) ? 1.0f : 0.0f;
  }
};

// The FUSED spelling: per shard, one kernel walks the shard's global index
// range, generates the sample pair positionally, tests and block-reduces —
// nothing is materialized. (The future generic form of this loop is a
// tabulate/transform_reduce over a lazy random input view.)
__global__ void fused_pi_kernel(
  unsigned long long offset, unsigned long long n, unsigned long long seed, unsigned long long* hits)
{
  const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  unsigned long long acc          = 0;
  for (unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride)
  {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, offset + i, 0, &st);
    const float x = curand_uniform(&st);
    const float y = curand_uniform(&st);
    acc += (x * x + y * y <= 1.0f) ? 1u : 0u;
  }
  __shared__ unsigned long long red[256];
  red[threadIdx.x] = acc;
  __syncthreads();
  for (unsigned off = blockDim.x / 2; off > 0; off /= 2)
  {
    if (threadIdx.x < off)
    {
      red[threadIdx.x] += red[threadIdx.x + off];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    atomicAdd(hits, red[0]);
  }
}

unsigned long long fused_hits(place_group& group, const ::std::vector<size_t>& shard_sizes, unsigned long long seed)
{
  unsigned long long* d_hits = static_cast<unsigned long long*>(group.memory_resource(0).allocate_sync(sizeof(*d_hits)));
  cuda_safe_call(cudaMemset(d_hits, 0, sizeof(*d_hits)));

  size_t offset = 0;
  for (size_t i = 0; i < shard_sizes.size(); i++)
  {
    const cudaStream_t s = group.get_stream(i);
    stream_scope scope(s);
    fused_pi_kernel<<<512, 256, 0, s>>>(offset, shard_sizes[i], seed, d_hits);
    cuda_safe_call(cudaGetLastError());
    offset += shard_sizes[i];
  }
  for (size_t i = 0; i < shard_sizes.size(); i++)
  {
    cuda_safe_call(cudaStreamSynchronize(group.get_stream(i)));
  }

  unsigned long long hits = 0;
  cuda_safe_call(cudaMemcpy(&hits, d_hits, sizeof(hits), cudaMemcpyDeviceToHost));
  group.memory_resource(0).deallocate_sync(d_hits, sizeof(*d_hits));
  return hits;
}

float estimate_hits(place_group& group, const ::std::vector<size_t>& shard_sizes)
{
  auto x      = sharded_array<float>::allocate(group, shard_sizes);
  auto y      = sharded_array<float>::allocate(group, shard_sizes);
  auto inside = sharded_array<float>::allocate(group, shard_sizes);

  generate_uniform(x, default_envs(x), /*seed=*/1234);
  generate_uniform(y, default_envs(y), /*seed=*/5678);

  zip_transform(inside, in_unit_circle{}, x, y);
  return reduce(inside, ::cuda::std::plus<float>{}, 0.0f);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  place_group group{make_locality_domain_grid()};
  const size_t n = 1u << 24;
  printf("sharded Monte Carlo pi: n = %zu samples over %zu places\n", n, group.size());

  // Sharding A: even split. Sharding B: deliberately uneven (odd boundary).
  ::std::vector<size_t> even(group.size(), n / group.size());
  even[0] += n % group.size();
  ::std::vector<size_t> uneven = even;
  if (group.size() > 1)
  {
    const size_t shift = 104729; // prime: odd boundary on purpose
    uneven[0] += shift;
    uneven[1] -= shift;
  }

  const float hits_a = estimate_hits(group, even);
  const float hits_b = estimate_hits(group, uneven);

  printf("  even sharding:   hits = %.0f  ->  pi ~ %.9f\n", hits_a, 4.0 * hits_a / n);
  printf("  uneven sharding: hits = %.0f  ->  pi ~ %.9f\n", hits_b, 4.0 * hits_b / n);

  if (hits_a != hits_b)
  {
    printf("FAILED: the materialized estimate is not sharding-invariant\n");
    return 1;
  }

  const unsigned long long fused_a = fused_hits(group, even, 4242);
  const unsigned long long fused_b = fused_hits(group, uneven, 4242);
  printf("  fused, even:     hits = %llu  ->  pi ~ %.9f   (zero bytes materialized)\n",
         fused_a,
         4.0 * static_cast<double>(fused_a) / n);
  printf("  fused, uneven:   hits = %llu  ->  pi ~ %.9f\n", fused_b, 4.0 * static_cast<double>(fused_b) / n);
  if (fused_a != fused_b)
  {
    printf("FAILED: the fused estimate is not sharding-invariant\n");
    return 1;
  }

  printf("both spellings are bitwise reproducible across shardings\n");
  return 0;
}
