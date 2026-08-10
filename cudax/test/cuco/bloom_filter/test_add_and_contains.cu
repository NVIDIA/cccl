//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Baseline round-trip coverage of the host-bulk API across a spread of policy configurations:
// an empty filter contains nothing, every added key is reported present, and `clear` empties the
// filter again.

// Temporary nvcc workaround __host__ __device__ dtor conflict in cuda::buffer
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/bloom_filter.cuh>
#include <cuda/experimental/__cuco/bloom_filter_policy.cuh>
#include <cuda/experimental/__cuco/hash_functions.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <class Key>
using hash64 = cudax::cuco::hash<Key, cudax::cuco::hash_algorithm::xxhash_64>;

template <class Key, class Policy>
struct config
{
  using key_type    = Key;
  using policy_type = Policy;
};

using configs = c2h::type_list<
  // Default policy: 8 x uint32_t words, fully horizontal add, fully vertical contains
  config<::cuda::std::int32_t, cudax::cuco::bloom_filter_policy<::cuda::std::int32_t>>,
  // Degenerate single-word block
  config<
    ::cuda::std::int32_t,
    cudax::cuco::
      bloom_filter_policy<::cuda::std::int32_t, hash64<::cuda::std::int32_t>, ::cuda::std::uint32_t, 1, 1, 1, 1, 1, 1>>,
  // Mixed layouts with a pattern-bit count that is not a multiple of the block width
  config<
    ::cuda::std::uint64_t,
    cudax::cuco::
      bloom_filter_policy<::cuda::std::uint64_t, hash64<::cuda::std::uint64_t>, ::cuda::std::uint32_t, 8, 12, 8, 1, 4, 2>>,
  // 64-bit words with a partially horizontal add
  config<float, cudax::cuco::bloom_filter_policy<float, hash64<float>, ::cuda::std::uint64_t, 4, 4, 2, 2, 1, 2>>,
  // Partially horizontal add with a fully vertical contains
  config<
    ::cuda::std::int32_t,
    cudax::cuco::
      bloom_filter_policy<::cuda::std::int32_t, hash64<::cuda::std::int32_t>, ::cuda::std::uint32_t, 8, 8, 2, 2, 1, 8>>,
  // 1-byte keys
  config<::cuda::std::uint8_t, cudax::cuco::bloom_filter_policy<::cuda::std::uint8_t>>,
  // 2-byte keys
  config<::cuda::std::uint16_t, cudax::cuco::bloom_filter_policy<::cuda::std::uint16_t>>,
  // The `conditional_add` and `early_exit_contains` optimizations enabled
  config<::cuda::std::int32_t,
         cudax::cuco::bloom_filter_policy<::cuda::std::int32_t,
                                          hash64<::cuda::std::int32_t>,
                                          ::cuda::std::uint32_t,
                                          8,
                                          8,
                                          8,
                                          1,
                                          8,
                                          1,
                                          cudax::cuco::conditional_add_mode::on,
                                          cudax::cuco::early_exit_contains_mode::on>>>;

template <class Key>
struct to_key
{
  __host__ __device__ Key operator()(int i) const noexcept
  {
    return static_cast<Key>(i);
  }
};

struct is_true
{
  __device__ bool operator()(bool v) const noexcept
  {
    return v;
  }
};

C2H_TEST("bloom_filter add and contains", "[bloom_filter]", configs)
{
  using cfg         = c2h::get<0, TestType>;
  using key_type    = typename cfg::key_type;
  using policy_type = typename cfg::policy_type;
  using filter_type =
    cudax::cuco::bloom_filter<key_type, ::cuda::std::dynamic_extent, ::cuda::thread_scope_device, policy_type>;

  // 1- and 2-byte key types cannot represent 400 distinct keys, so scale by the type's range
  constexpr int num_keys                   = (::cuda::std::numeric_limits<key_type>::max() > 400) ? 400 : 100;
  constexpr ::cuda::std::size_t num_blocks = 1000;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  filter_type filter{stream, mr, num_blocks};

  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(filter.block_extent()), typename filter_type::size_type>);
  REQUIRE(filter.block_extent() == num_blocks);
  REQUIRE(filter.num_words() == num_blocks * static_cast<::cuda::std::size_t>(filter_type::words_per_block));

  const auto keys_begin = cuda::transform_iterator(cuda::counting_iterator<int>{0}, to_key<key_type>{});
  const auto keys_end   = keys_begin + num_keys;

  auto contained = ::cuda::make_buffer<bool>(stream, mr, num_keys, false);

  SECTION("Non-added keys are not contained")
  {
    filter.contains(stream, keys_begin, keys_end, contained.begin());
    REQUIRE(::thrust::none_of(
      ::thrust::cuda::par.on(stream.get()), contained.data(), contained.data() + num_keys, is_true{}));
  }

  SECTION("All added keys are contained")
  {
    filter.add(stream, keys_begin, keys_end);
    filter.contains(stream, keys_begin, keys_end, contained.begin());
    REQUIRE(
      ::thrust::all_of(::thrust::cuda::par.on(stream.get()), contained.data(), contained.data() + num_keys, is_true{}));
  }

  SECTION("After clearing the filter no keys are contained")
  {
    filter.add(stream, keys_begin, keys_end);
    filter.clear(stream);
    filter.contains(stream, keys_begin, keys_end, contained.begin());
    REQUIRE(::thrust::none_of(
      ::thrust::cuda::par.on(stream.get()), contained.data(), contained.data() + num_keys, is_true{}));
  }

  SECTION("Empty ranges are a no-op")
  {
    filter.add(stream, keys_begin, keys_begin);
    filter.contains(stream, keys_begin, keys_end, contained.begin());
    REQUIRE(::thrust::none_of(
      ::thrust::cuda::par.on(stream.get()), contained.data(), contained.data() + num_keys, is_true{}));
  }
}

C2H_TEST("bloom_filter static block count", "[bloom_filter]")
{
  using key_type                           = ::cuda::std::int32_t;
  constexpr ::cuda::std::size_t num_blocks = 1000;
  using filter_type                        = cudax::cuco::bloom_filter<key_type, num_blocks>;
  constexpr int num_keys                   = 400;

  STATIC_REQUIRE(filter_type::num_blocks_v == num_blocks);

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  filter_type filter{stream, mr};
  REQUIRE(filter.block_extent() == num_blocks);

  const auto keys_begin = cuda::transform_iterator(cuda::counting_iterator<int>{0}, to_key<key_type>{});
  const auto keys_end   = keys_begin + num_keys;

  auto contained = ::cuda::make_buffer<bool>(stream, mr, num_keys, false);
  filter.add(stream, keys_begin, keys_end);
  filter.contains(stream, keys_begin, keys_end, contained.begin());
  REQUIRE(
    ::thrust::all_of(::thrust::cuda::par.on(stream.get()), contained.data(), contained.data() + num_keys, is_true{}));
}
