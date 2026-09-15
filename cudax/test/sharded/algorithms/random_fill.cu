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
 * @brief Sharding-invariance certification of the cuRAND fill tier
 *        (`random.cuh`): for {uniform, normal} x {float, double}, the fill
 *        of a sharded view is BITWISE identical to the whole-array reference
 *        for every sharding — one shard, even halves, odd/prime cuts, and
 *        per-domain lanes. This bitwise A/B is the contract's gate: it must
 *        be re-run when the toolkit changes.
 */

#include <cuda/experimental/__sharded/random.cuh> // opt-in vendor tier
#include <cuda/experimental/sharded.cuh>

#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
constexpr size_t N                = 1u << 22;
constexpr unsigned long long SEED = 0xC0FFEE;

// Whole-array references defining each path's sequence. FLOAT: the stock
// host API (the float path is a bitwise drop-in for it). DOUBLE: the
// positional device kernel at offset 0 (the sequence this interface defines;
// see random.cuh's path note).
template <typename T>
void reference_fill(T* d, bool normal)
{
  if constexpr (::cuda::std::is_same_v<T, double>)
  {
    if (normal)
    {
      reserved::__launch_positional_fill_double<true>(d, N, 0, SEED, 0.0, 1.0, nullptr);
    }
    else
    {
      reserved::__launch_positional_fill_double<false>(d, N, 0, SEED, 0.0, 0.0, nullptr);
    }
    cuda_safe_call(cudaDeviceSynchronize());
  }
  else
  {
    curandGenerator_t g;
    reserved::__curand_check(curandCreateGenerator(&g, CURAND_RNG_PSEUDO_PHILOX4_32_10), "ref create");
    reserved::__curand_check(curandSetGeneratorOrdering(g, CURAND_ORDERING_PSEUDO_DEFAULT), "ref ordering");
    reserved::__curand_check(curandSetPseudoRandomGeneratorSeed(g, SEED), "ref seed");
    if (normal)
    {
      reserved::__curand_check(curandGenerateNormal(g, d, N, 0.f, 1.f), "ref generate");
    }
    else
    {
      reserved::__curand_check(curandGenerateUniform(g, d, N), "ref generate");
    }
    cuda_safe_call(cudaDeviceSynchronize());
    curandDestroyGenerator(g);
  }
}

template <typename T>
int run_case(const char* name, place_group& group, bool normal)
{
  int failures = 0;

  T* d_ref = nullptr;
  cuda_safe_call(cudaMalloc(&d_ref, N * sizeof(T)));
  reference_fill<T>(d_ref, normal);
  std::vector<T> ref(N);
  cuda_safe_call(cudaMemcpy(ref.data(), d_ref, N * sizeof(T), cudaMemcpyDeviceToHost));
  cuda_safe_call(cudaFree(d_ref));

  // Shardings as interior cut points (0 and N implicit). Odd boundaries are
  // deliberate: the contract holds at ANY cut.
  std::vector<std::pair<const char*, std::vector<size_t>>> shardings = {
    {"1 shard", {}},
    {"even halves", {N / 2}},
    {"prime cuts", {104729, 1299709, 2750159}},
    // Size-1 shards at odd offsets: pins the normal path's even-count
    // boundary decomposition (head-only, tail-only shards).
    {"tiny + odd", {1, 2, 3, 104730}},
  };
  {
    std::vector<size_t> cuts;
    for (size_t i = 1; i < group.size(); i++)
    {
      cuts.push_back(i * (N / group.size()) + 1);
    }
    shardings.emplace_back("per-domain", std::move(cuts));
  }

  T* d_buf = nullptr;
  cuda_safe_call(cudaMalloc(&d_buf, N * sizeof(T)));
  std::vector<T> got(N);

  for (const auto& [sname, cuts] : shardings)
  {
    cuda_safe_call(cudaMemset(d_buf, 0, N * sizeof(T)));

    std::vector<cuda::std::span<T>> pieces;
    size_t prev = 0;
    for (size_t c : cuts)
    {
      pieces.push_back({d_buf + prev, c - prev});
      prev = c;
    }
    pieces.push_back({d_buf + prev, N - prev});
    const auto view = make_sharded_view(pieces);

    // One env per shard, shard i on domain (i mod P), distinct lanes.
    std::vector<decltype(place_group::env(std::declval<const cuda::experimental::places::data_place&>(), cudaStream_t{}))>
      envs;
    for (size_t i = 0; i < pieces.size(); i++)
    {
      const size_t p = i % group.size();
      envs.push_back(place_group::env(group.place(p).affine_data_place(), group.get_stream(p, i / group.size())));
    }

    if (normal)
    {
      generate_normal(view, envs, SEED, T(0), T(1));
    }
    else
    {
      generate_uniform(view, envs, SEED);
    }

    cuda_safe_call(cudaMemcpy(got.data(), d_buf, N * sizeof(T), cudaMemcpyDeviceToHost));
    const bool ok = std::memcmp(got.data(), ref.data(), N * sizeof(T)) == 0;
    printf("  %-12s %-12s %zu shards: %s\n", name, sname, pieces.size(), ok ? "bitwise match" : "MISMATCH");
    failures += !ok;
  }

  cuda_safe_call(cudaFree(d_buf));
  return failures;
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  place_group group{make_locality_domain_grid()};
  printf("random_fill invariance: %zu places, N=%zu\n", group.size(), N);

  int failures = 0;
  failures += run_case<float>("uniform f32", group, false);
  failures += run_case<double>("uniform f64", group, false);
  failures += run_case<float>("normal f32", group, true);
  failures += run_case<double>("normal f64", group, true);

  if (failures != 0)
  {
    printf("random_fill: FAILED (%d mismatches)\n", failures);
    return 1;
  }
  printf("random_fill: all shardings bitwise-invariant\n");
  return 0;
}
