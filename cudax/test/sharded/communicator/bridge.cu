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
 * @brief THE BRIDGE TEST: the MGMN range algorithm `cuda::experimental::reduce`
 *        running over places communicators and environments manufactured from
 *        a sharded container by `bind_engine`, on locality domains, checked
 *        against a single-place CUB reference. This is the conformance
 *        evidence as CI: sharded containers drive the MGMN constructs
 *        unmodified, through both combine paths (all_reduce and
 *        all_gather-plus-local-combine).
 */

#include <cub/device/device_reduce.cuh>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

namespace cudax = cuda::experimental;
using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place_scope;
using cuda::experimental::places::place_group;

namespace
{
uint32_t float_bits(float v)
{
  uint32_t b;
  ::std::memcpy(&b, &v, sizeof(b));
  return b;
}

template <typename Comm>
::std::vector<uint32_t> run_bridge_reduce(place_group& group, sharded_array<float>& data, int repetitions)
{
  const size_t nranks = group.size();

  // The container tier manufactures everything the engine consumes.
  auto b = bind_engine<Comm>(group, data);

  // One output element per rank, on the rank's place.
  ::std::vector<float*> out(nranks);
  for (size_t r = 0; r < nranks; r++)
  {
    auto dp = group.place(r).affine_data_place();
    exec_place_scope scope(group.place(r));
    out[r] = static_cast<float*>(dp.allocate(sizeof(float), group.get_stream(r)));
  }

  ::std::vector<uint32_t> bits;
  for (int rep = 0; rep < repetitions; rep++)
  {
    cudax::reduce(cudax::broadcasted, b.comms, b.envs, b.shard_data, b.shard_sizes, out, 0.0f, cuda::std::plus<>{},
                  0.0f);

    // Read back every rank's result; the broadcasted policy makes them all
    // carry the same value, which we check bitwise.
    ::std::vector<float> results(nranks);
    for (size_t r = 0; r < nranks; r++)
    {
      exec_place_scope scope(group.place(r));
      cuda_safe_call(cudaMemcpyAsync(&results[r], out[r], sizeof(float), cudaMemcpyDeviceToHost, group.get_stream(r)));
      cuda_safe_call(cudaStreamSynchronize(group.get_stream(r)));
    }
    for (size_t r = 1; r < nranks; r++)
    {
      EXPECT(float_bits(results[r]) == float_bits(results[0]));
    }
    bits.push_back(float_bits(results[0]));
  }

  for (size_t r = 0; r < nranks; r++)
  {
    auto dp = group.place(r).affine_data_place();
    exec_place_scope scope(group.place(r));
    dp.deallocate(out[r], sizeof(float), group.get_stream(r));
    cuda_safe_call(cudaStreamSynchronize(group.get_stream(r)));
  }

  return bits;
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  // Uneven shards: the per-shard size range must reach the engine as-is.
  const size_t n = (1 << 22) + 12345;
  ::std::vector<size_t> sizes(group.size(), n / (2 * group.size()));
  sizes[0] = n - (group.size() - 1) * (n / (2 * group.size()));

  ::std::vector<float> host(n);
  {
    ::std::mt19937 rng(7);
    ::std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : host)
    {
      v = dist(rng);
    }
  }
  double ref = 0.0;
  for (float v : host)
  {
    ref += static_cast<double>(v);
  }
  const double tol = 1e-6 * ref;

  auto data = sharded_array<float>::allocate(group, sizes);
  data.copy_from_host(host.data());

  // Single-place CUB reference over the same values, contiguous.
  float cub_result = 0.0f;
  {
    float *d_in = nullptr, *d_out = nullptr;
    cuda_safe_call(cudaMalloc(&d_in, n * sizeof(float)));
    cuda_safe_call(cudaMalloc(&d_out, sizeof(float)));
    cuda_safe_call(cudaMemcpy(d_in, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    void* d_temp      = nullptr;
    size_t temp_bytes = 0;
    cuda_safe_call(cub::DeviceReduce::Reduce(nullptr, temp_bytes, d_in, d_out, n, cuda::std::plus<>{}, 0.0f));
    cuda_safe_call(cudaMalloc(&d_temp, temp_bytes));
    cuda_safe_call(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out, n, cuda::std::plus<>{}, 0.0f));
    cuda_safe_call(cudaMemcpy(&cub_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    cuda_safe_call(cudaFree(d_temp));
    cuda_safe_call(cudaFree(d_out));
    cuda_safe_call(cudaFree(d_in));
    EXPECT(::std::abs(static_cast<double>(cub_result) - ref) <= tol);
  }

  // Direct combine path (all_reduce) + bit-determinism over 5 runs.
  {
    auto bits = run_bridge_reduce<places_communicator>(group, data, 5);
    float v;
    ::std::memcpy(&v, &bits[0], sizeof(v));
    EXPECT(::std::abs(static_cast<double>(v) - ref) <= tol);
    EXPECT(::std::abs(static_cast<double>(v) - static_cast<double>(cub_result)) <= tol);
    for (size_t i = 1; i < bits.size(); i++)
    {
      EXPECT(bits[i] == bits[0]); // fixed fold order: bit-identical runs
    }
  }

  // Gather combine path (communicator without all_reduce) + determinism.
  {
    auto bits = run_bridge_reduce<basic_places_communicator>(group, data, 5);
    float v;
    ::std::memcpy(&v, &bits[0], sizeof(v));
    EXPECT(::std::abs(static_cast<double>(v) - ref) <= tol);
    EXPECT(::std::abs(static_cast<double>(v) - static_cast<double>(cub_result)) <= tol);
    for (size_t i = 1; i < bits.size(); i++)
    {
      EXPECT(bits[i] == bits[0]);
    }
  }

  return 0;
}
