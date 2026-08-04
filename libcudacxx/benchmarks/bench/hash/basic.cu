//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

#include <nvbench/nvbench.cuh>
#include <nvbench/range.cuh>

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

template <typename Result>
__device__ __forceinline__ cuda::std::uint64_t reduce_hash_result(Result result) noexcept
{
  if constexpr (sizeof(Result) <= sizeof(cuda::std::uint64_t))
  {
    return static_cast<cuda::std::uint64_t>(result);
  }
  else
  {
    static_assert(sizeof(Result) == 4 * sizeof(cuda::std::uint32_t));
    const auto words = cuda::std::bit_cast<cuda::std::array<cuda::std::uint32_t, 4>>(result);
    return static_cast<cuda::std::uint64_t>(words[0]) + static_cast<cuda::std::uint64_t>(words[1])
         + static_cast<cuda::std::uint64_t>(words[2]) + static_cast<cuda::std::uint64_t>(words[3]);
  }
}

template <cuda::std::int32_t BlockSize, typename Key, typename Hasher, typename OutputIt>
__global__ void hash_bench_kernel(Hasher hash, cuda::std::size_t n, OutputIt out, bool materialize_result)
{
  const cuda::std::size_t gid         = static_cast<cuda::std::size_t>(BlockSize) * blockIdx.x + threadIdx.x;
  const cuda::std::size_t loop_stride = static_cast<cuda::std::size_t>(gridDim.x) * BlockSize;
  cuda::std::size_t idx               = gid;

  cuda::std::uint64_t agg{};

  while (idx < n)
  {
    const Key key(static_cast<cuda::std::int32_t>(idx));
    for (cuda::std::int32_t i = 0; i < n_repeats; ++i)
    {
      agg += reduce_hash_result(hash(key));
    }
    idx += loop_stride;
  }

  if (materialize_result)
  {
    out[gid] = agg;
  }
}

// benchmark evaluating performance of various hash functions
template <typename HasherTag, typename Key>
void hash_eval(nvbench::state& state, nvbench::type_list<HasherTag, Key>)
{
  using hash_t = typename HasherTag::template fn<Key>;

  const bool materialize_result = false;
  constexpr auto block_size     = 128;
  const auto num_keys           = state.get_int64("NumInputs");
  const auto grid_size          = (num_keys + block_size * 16 - 1) / block_size * 16;

  thrust::device_vector<cuda::std::uint64_t> hash_values((materialize_result) ? num_keys : 1);

  state.add_element_count(num_keys);

  state.exec([&](nvbench::launch& launch) {
    hash_bench_kernel<block_size, Key>
      <<<grid_size, block_size, 0, launch.get_stream()>>>(hash_t{}, num_keys, hash_values.begin(), materialize_result);
  });
}

struct xxhash_32_tag
{
  template <typename Key>
  using fn = cuda::hash<Key, cuda::hash_algorithm::xxhash_32>;
};

struct xxhash_64_tag
{
  template <typename Key>
  using fn = cuda::hash<Key, cuda::hash_algorithm::xxhash_64>;
};

struct murmurhash3_32_tag
{
  template <typename Key>
  using fn = cuda::hash<Key, cuda::hash_algorithm::murmurhash3_32>;
};

struct murmurhash3_x86_128_tag
{
  template <typename Key>
  using fn = cuda::hash<Key, cuda::hash_algorithm::murmurhash3_x86_128>;
};

struct murmurhash3_x64_128_tag
{
  template <typename Key>
  using fn = cuda::hash<Key, cuda::hash_algorithm::murmurhash3_x64_128>;
};

NVBENCH_BENCH_TYPES(
  hash_eval,
  NVBENCH_TYPE_AXES(
    nvbench::
      type_list<xxhash_32_tag, xxhash_64_tag, murmurhash3_32_tag, murmurhash3_x86_128_tag, murmurhash3_x64_128_tag>,
    nvbench::type_list<cuda::std::int32_t, large_key<4>, large_key<8>, large_key<16>, large_key<32>>))
  .set_name("hash_function_eval")
  .set_type_axes_names({"Hash", "Key"})
  .add_int64_power_of_two_axis("NumInputs", nvbench::range(18, 26, 4));
