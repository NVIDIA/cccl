// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_vector.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <nvbench/nvbench.cuh>
#include <nvbench/range.cuh>

namespace cudax = cuda::experimental;

// repeat hash computation n times
static constexpr auto n_repeats = 100;

template <cuda::std::int32_t Words>
struct large_key
{
  constexpr __host__ __device__ large_key(cuda::std::int32_t seed) noexcept
  {
    for (cuda::std::int32_t i = 0; i < Words; ++i)
    {
      data_[i] = seed;
    }
  }

private:
  cuda::std::int32_t data_[Words];
};

template <cuda::std::int32_t BlockSize, typename Key, typename Hasher, typename OutputIt>
__global__ void hash_bench_kernel(Hasher hash, size_t n, OutputIt out, bool materialize_result)
{
  size_t const gid         = BlockSize * blockIdx.x + threadIdx.x;
  size_t const loop_stride = gridDim.x * BlockSize;
  size_t idx               = gid;
  using result_t           = decltype(hash(0));

  result_t agg{};

  while (idx < n)
  {
    Key key(idx);
    for (cuda::std::int32_t i = 0; i < n_repeats; ++i)
    { // execute hash func n times
      agg += hash(key);
    }
    idx += loop_stride;
  }

  if (materialize_result)
  {
    out[gid] = agg;
  }
}

// benchmark evaluating performance of various hash functions
template <typename StrategyTag, typename Key>
void hash_eval(nvbench::state& state, nvbench::type_list<StrategyTag, Key>)
{
  using Hash = typename StrategyTag::template fn<Key>;

  bool const materialize_result = false;
  constexpr auto block_size     = 128;
  auto const num_keys           = state.get_int64("NumInputs");
  auto const grid_size          = (num_keys + block_size * 16 - 1) / block_size * 16;
  using result_t                = decltype(std::declval<Hash>()(std::declval<cuda::std::int32_t>()));

  thrust::device_vector<result_t> hash_values((materialize_result) ? num_keys : 1);

  state.add_element_count(num_keys);

  state.exec([&](nvbench::launch& launch) {
    hash_bench_kernel<block_size, Key>
      <<<grid_size, block_size, 0, launch.get_stream()>>>(Hash{}, num_keys, hash_values.begin(), materialize_result);
  });
}

struct xxhash_32_tag
{
  template <typename Key>
  using fn = cudax::cuco::Hash<Key, cudax::cuco::HashStrategy::XXHash_32>;
};

NVBENCH_BENCH_TYPES(
  hash_eval,
  NVBENCH_TYPE_AXES(nvbench::type_list<xxhash_32_tag>,
                    nvbench::type_list<cuda::std::int32_t, large_key<4>, large_key<8>, large_key<16>, large_key<32>>))
  .set_name("hash_function_eval")
  .set_type_axes_names({"Hash", "Key"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(18, 26, 4));
