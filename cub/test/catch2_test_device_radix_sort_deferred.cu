// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cuda/argument>
#include <cuda/execution>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstdint>

#include "catch2_large_array_sort_helper.cuh"
#include "catch2_radix_sort_helper.cuh"
#include "cub_test_macros.h"
#include <c2h/catch2_allocation_failure.h>
#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <c2h/generators.h>

template <int TilesPerBatch = 3>
struct small_batch_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr cub::RadixSortPolicy operator()(cuda::compute_capability cc) const
  {
    auto policy = cub::detail::radix_sort::policy_selector_from_types<uint32_t, uint32_t, uint64_t, true>{}(cc);
    policy.histogram.private_partitions = 1;
    policy.onesweep.threads_per_block   = 128;
    policy.onesweep.items_per_thread    = 4;
    policy.onesweep.tiles_per_batch     = TilesPerBatch;
    return policy;
  }
};

using deferred_radix_count_types      = c2h::type_list<int32_t, uint32_t, int64_t, uint64_t>;
using deferred_radix_scalar_key_types = c2h::type_list<int32_t, int64_t, float, double>;

using deferred_byte_policy =
  cub::detail::radix_sort::policy_selector_from_types<uint8_t, cub::NullType, uint64_t, true>;
using deferred_short_pair_policy =
  cub::detail::radix_sort::policy_selector_from_types<uint16_t, uint32_t, uint32_t, true>;

static_assert(deferred_byte_policy{}(cuda::compute_capability{7, 5}).algorithm == cub::RadixSortAlgorithm::onesweep);
static_assert(deferred_byte_policy{}(cuda::compute_capability{8, 0}).algorithm == cub::RadixSortAlgorithm::onesweep);
static_assert(deferred_short_pair_policy{}(cuda::compute_capability{8, 0}).algorithm
              == cub::RadixSortAlgorithm::onesweep);

using deferred_int_policy = cub::detail::radix_sort::policy_selector_from_types<int32_t, cub::NullType, uint64_t, true>;
using immediate_int_policy  = cub::detail::radix_sort::policy_selector_from_types<int32_t, cub::NullType, uint64_t>;
using deferred_pair_policy  = cub::detail::radix_sort::policy_selector_from_types<int32_t, uint32_t, uint64_t, true>;
using immediate_pair_policy = cub::detail::radix_sort::policy_selector_from_types<int32_t, uint32_t, uint64_t>;
using deferred_narrow_policy =
  cub::detail::radix_sort::policy_selector_from_types<int32_t, cub::NullType, uint32_t, true>;
using immediate_narrow_policy = cub::detail::radix_sort::policy_selector_from_types<int32_t, cub::NullType, uint32_t>;

static_assert(deferred_int_policy{}(cuda::compute_capability{8, 0}).onesweep.tiles_per_batch == 4096);
static_assert(deferred_int_policy{}(cuda::compute_capability{8, 0})
              == immediate_int_policy{}(cuda::compute_capability{8, 0}));
static_assert(deferred_int_policy{}(cuda::compute_capability{12, 0})
              == immediate_int_policy{}(cuda::compute_capability{12, 0}));
static_assert(deferred_pair_policy{}(cuda::compute_capability{10, 0})
              == immediate_pair_policy{}(cuda::compute_capability{10, 0}));
static_assert(deferred_narrow_policy{}(cuda::compute_capability{10, 0})
              == immediate_narrow_policy{}(cuda::compute_capability{10, 0}));

template <typename KeyT>
struct blackwell_deferred_radix_test_policy
{
  _CCCL_HOST_DEVICE_API constexpr cub::RadixSortPolicy operator()(cuda::compute_capability) const
  {
    auto policy = cub::detail::radix_sort::policy_selector_from_types<KeyT, cub::NullType, uint64_t, true>{}(
      cuda::compute_capability{10, 3});
    policy.onesweep.tiles_per_batch = 3;
    return policy;
  }
};

CUB_TEST("deferred tuned policies sort across small batch windows",
         "[device][radix_sort][deferred][policy]",
         CUB_SMALL,
         deferred_radix_scalar_key_types)
{
  using key_t           = c2h::get<0, TestType>;
  constexpr auto policy = blackwell_deferred_radix_test_policy<key_t>{}(cuda::compute_capability{});
  // Force the new tile policies on the test GPU. Three-slot windows exercise
  // both lookback buffers, reuse, and a partial final batch without huge inputs.
  constexpr int n = 9 * policy.onesweep.threads_per_block * policy.onesweep.items_per_thread + 17;
  c2h::device_vector<key_t> first(n, thrust::no_init), second(n, thrust::no_init);
  c2h::gen(C2H_SEED(1), first, key_t{-1000}, key_t{1000});
  const auto expected = radix_sort_reference(first, false);
  c2h::device_vector<uint64_t> device_count(1, n);
  cub::DoubleBuffer<key_t> keys(first.data().get(), second.data().get());
  const auto count = cuda::args::deferred{device_count.data().get()};
  const auto env   = cuda::execution::tune(blackwell_deferred_radix_test_policy<key_t>{});
  size_t bytes{};
  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, 0, sizeof(key_t) * 8, env));
  c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
  REQUIRE(
    cudaSuccess == cub::DeviceRadixSort::SortKeys(scratch.data().get(), bytes, keys, count, 0, sizeof(key_t) * 8, env));
  const c2h::host_vector<key_t> output = keys.selector ? second : first;
  REQUIRE(output == expected);
}

CUB_TEST("deferred radix sort reuses a small window and preserves pair stability",
         "[device][radix_sort][deferred]",
         CUB_SMALL)
{
  using key_t           = uint32_t;
  using count_t         = uint64_t;
  constexpr auto policy = small_batch_policy_selector<>{}(cuda::compute_capability{7, 5});
  constexpr int tile    = policy.onesweep.threads_per_block * policy.onesweep.items_per_thread;
  constexpr int batch   = tile * policy.onesweep.tiles_per_batch;
  // The first two batches use the initially cleared buffers; the third reuses
  // the first buffer. Cover both boundaries and a partial final batch.
  const int n = GENERATE_COPY(
    0,
    1,
    tile - 1,
    tile,
    tile + 1,
    batch - 1,
    batch,
    batch + 1,
    2 * batch,
    2 * batch + 1,
    3 * batch,
    3 * batch + 1,
    11 * batch + 23);
  const bool descending = GENERATE(false, true);
  const int end_bit     = GENERATE(0, 7, 16, int(sizeof(key_t) * 8));
  const int begin_bit   = end_bit == 0 ? 0 : 3;
  const bool constant   = GENERATE(false, true);
  CAPTURE(n, descending, begin_bit, end_bit, constant);

  c2h::device_vector<key_t> first(n, thrust::no_init), second(n, thrust::no_init);
  c2h::gen(C2H_SEED(1), first, key_t{0}, constant ? key_t{0} : (cuda::std::numeric_limits<key_t>::max)());
  c2h::device_vector<uint32_t> values_first(n, thrust::no_init), values_second(n, thrust::no_init);
  thrust::sequence(c2h::device_policy, values_first.begin(), values_first.end());
  const auto expected = radix_sort_reference(first, values_first, descending, begin_bit, end_bit);
  c2h::device_vector<count_t> device_count(1, static_cast<count_t>(n));
  cub::DoubleBuffer<key_t> keys(first.data().get(), second.data().get());
  cub::DoubleBuffer<uint32_t> pairs(values_first.data().get(), values_second.data().get());
  const auto count = cuda::args::deferred{device_count.data().get()};
  const auto env   = cuda::execution::tune(small_batch_policy_selector<>{});
  size_t bytes{};
  REQUIRE(cudaSuccess
          == (descending
                ? cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, keys, pairs, count, begin_bit, end_bit, env)
                : cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, pairs, count, begin_bit, end_bit, env)));
  REQUIRE(keys.selector == 0);
  REQUIRE(pairs.selector == 0);
  c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
  REQUIRE(
    cudaSuccess
    == (descending
          ? cub::DeviceRadixSort::SortPairsDescending(
              scratch.data().get(), bytes, keys, pairs, count, begin_bit, end_bit, env)
          : cub::DeviceRadixSort::SortPairs(scratch.data().get(), bytes, keys, pairs, count, begin_bit, end_bit, env)));
  const int passes = cuda::ceil_div(end_bit - begin_bit, policy.onesweep.radix_bits);
  REQUIRE(keys.selector == (passes % 2));
  REQUIRE(pairs.selector == keys.selector);

  const c2h::host_vector<key_t> output           = keys.selector ? second : first;
  const c2h::host_vector<uint32_t> output_values = pairs.selector ? values_second : values_first;
  REQUIRE(output == expected.first);
  REQUIRE(output_values == expected.second);
}

CUB_TEST("deferred radix queries do not read the count and allocate fixed bookkeeping",
         "[device][radix_sort][deferred]",
         CUB_SMALL)
{
  cub::DoubleBuffer<uint32_t> keys;
  const auto unreadable = cuda::args::deferred{static_cast<int64_t*>(nullptr)};
  size_t bytes{};
  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, unreadable));
  REQUIRE(bytes > 0);
  REQUIRE(keys.selector == 0);
  const auto bounded =
    cuda::args::deferred{static_cast<int64_t*>(nullptr), cuda::args::bounds<int64_t{0}, int64_t{1024}>()};
  size_t bounded_bytes{};
  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(nullptr, bounded_bytes, keys, bounded));
  REQUIRE(bounded_bytes == bytes);
}

CUB_TEST("bounded deferred radix preserves separate input and sorts only N output items",
         "[device][radix_sort][deferred][bounded]",
         CUB_SMALL,
         deferred_radix_scalar_key_types)
{
  using key_t           = c2h::get<0, TestType>;
  constexpr auto policy = small_batch_policy_selector<>{}(cuda::compute_capability{7, 5});
  constexpr int64_t batch_items =
    policy.onesweep.tiles_per_batch * policy.onesweep.threads_per_block * policy.onesweep.items_per_thread;
  constexpr int64_t capacity = 7 * batch_items + 19;
  const int64_t n            = GENERATE_COPY(int64_t{0}, int64_t{1}, batch_items + 1, capacity - 11);
  const bool descending      = GENERATE(false, true);
  const bool pairs           = GENERATE(false, true);
  const int end_bit          = GENERATE(0, 7, 16, 24, int(sizeof(key_t) * 8));
  const int begin_bit        = end_bit == 0 ? 0 : 3;
  CAPTURE(n, descending, pairs, begin_bit, end_bit);
  c2h::device_vector<key_t> input(capacity, thrust::no_init), output(capacity, key_t{42});
  c2h::gen(C2H_SEED(1), input, key_t{-1000}, key_t{1000});
  const c2h::host_vector<key_t> original = input;
  c2h::device_vector<uint32_t> values_in(capacity, thrust::no_init), values_out(capacity, 0xabcdefu);
  thrust::sequence(c2h::device_policy, values_in.begin(), values_in.end());
  const c2h::host_vector<uint32_t> original_values = values_in;
  const c2h::device_vector<key_t> reference_keys(input.begin(), input.begin() + n);
  const c2h::device_vector<uint32_t> reference_values(values_in.begin(), values_in.begin() + n);
  const auto expected = radix_sort_reference(reference_keys, reference_values, descending, begin_bit, end_bit);
  c2h::device_vector<int64_t> device_count(1, n);
  const auto env   = cuda::execution::tune(small_batch_policy_selector<>{});
  const auto count = cuda::args::deferred{device_count.data().get(), cuda::args::bounds(int64_t{0}, capacity)};
  auto* keys_in    = input.data().get();
  auto* keys_out   = output.data().get();
  auto* pairs_in   = values_in.data().get();
  auto* pairs_out  = values_out.data().get();
  auto sort        = [&](void* scratch, size_t& bytes) {
    if (pairs)
    {
      return descending
             ? cub::DeviceRadixSort::SortPairsDescending(
                 scratch, bytes, keys_in, keys_out, pairs_in, pairs_out, count, begin_bit, end_bit, env)
             : cub::DeviceRadixSort::SortPairs(
                 scratch, bytes, keys_in, keys_out, pairs_in, pairs_out, count, begin_bit, end_bit, env);
    }
    return descending
           ? cub::DeviceRadixSort::SortKeysDescending(scratch, bytes, keys_in, keys_out, count, begin_bit, end_bit, env)
           : cub::DeviceRadixSort::SortKeys(scratch, bytes, keys_in, keys_out, count, begin_bit, end_bit, env);
  };
  size_t bytes{};
  REQUIRE(cudaSuccess == sort(nullptr, bytes));
  c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
  REQUIRE(cudaSuccess == sort(scratch.data().get(), bytes));
  const c2h::host_vector<key_t> result = output;
  REQUIRE(std::equal(expected.first.begin(), expected.first.end(), result.begin()));
  REQUIRE(std::all_of(result.begin() + n, result.end(), [](key_t value) {
    return value == key_t{42};
  }));
  REQUIRE(original == input);
  REQUIRE(original_values == values_in);
  if (pairs)
  {
    const c2h::host_vector<uint32_t> result_values = values_out;
    REQUIRE(std::equal(expected.second.begin(), expected.second.end(), result_values.begin()));
    REQUIRE(std::all_of(result_values.begin() + n, result_values.end(), [](uint32_t value) {
      return value == 0xabcdefu;
    }));
  }
}

CUB_TEST("separate deferred radix storage queries require bounds and size only intermediate arrays",
         "[device][radix_sort][deferred][bounded]",
         CUB_SMALL,
         deferred_radix_count_types)
{
  using count_t        = c2h::get<0, TestType>;
  uint32_t* keys       = nullptr;
  uint64_t* values     = nullptr;
  const auto env       = cuda::execution::tune(small_batch_policy_selector<>{});
  const auto unbounded = cuda::args::deferred{static_cast<count_t*>(nullptr)};
  size_t bytes{};
  REQUIRE(cudaErrorInvalidValue == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, keys, unbounded, 0, 32, env));
  REQUIRE(cudaErrorInvalidValue
          == cub::DeviceRadixSort::SortKeysDescending(nullptr, bytes, keys, keys, unbounded, 0, 32, env));
  REQUIRE(cudaErrorInvalidValue
          == cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, keys, values, values, unbounded, 0, 32, env));
  REQUIRE(
    cudaErrorInvalidValue
    == cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, keys, keys, values, values, unbounded, 0, 32, env));
  if constexpr (cuda::std::is_signed_v<count_t>)
  {
    const auto negative =
      cuda::args::deferred{static_cast<count_t*>(nullptr), cuda::args::bounds(count_t{-2}, count_t{-1})};
    REQUIRE(cudaErrorInvalidValue == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, keys, negative, 0, 32, env));
  }
  if constexpr (sizeof(count_t) == 8)
  {
    const auto overflowing = cuda::args::deferred{
      static_cast<count_t*>(nullptr), cuda::args::bounds(count_t{0}, (cuda::std::numeric_limits<count_t>::max)() - 1)};
    REQUIRE(cudaErrorInvalidValue
            == cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, keys, values, values, overflowing, 0, 32, env));
  }
  auto query = [&](count_t capacity, bool pairs, int end_bit) {
    // Null data and count pointers must remain unread during a storage query.
    const auto count = cuda::args::deferred{static_cast<count_t*>(nullptr), cuda::args::bounds(count_t{0}, capacity)};
    size_t queried{};
    REQUIRE(
      cudaSuccess
      == (pairs ? cub::DeviceRadixSort::SortPairs(nullptr, queried, keys, keys, values, values, count, 0, end_bit, env)
                : cub::DeviceRadixSort::SortKeys(nullptr, queried, keys, keys, count, 0, end_bit, env)));
    return queried;
  };
  for (bool pairs : {false, true})
  {
    // These capacities are multiples of the temporary storage alignment.
    const size_t item_bytes = sizeof(uint32_t) + (pairs ? sizeof(uint64_t) : 0);
    REQUIRE(query(2048, pairs, 32) - query(1024, pairs, 32) == 1024 * item_bytes);
    REQUIRE(query(2048, pairs, 7) == query(1024, pairs, 7));
    REQUIRE(query(2048, pairs, 0) == query(1024, pairs, 0));
    REQUIRE(query(0, pairs, 32) > 0);
    if constexpr (sizeof(count_t) == 8)
    {
      REQUIRE(query((count_t{1} << 32) + 1024, pairs, 32) - query(1024, pairs, 32) == (uint64_t{1} << 32) * item_bytes);
    }
  }
}

CUB_TEST("deferred radix graph replays with growing and shrinking counts",
         "[device][radix_sort][deferred][graph]",
         CUB_SMALL)
{
  constexpr auto policy = small_batch_policy_selector<>{}(cuda::compute_capability{7, 5});
  constexpr int batch_items =
    policy.onesweep.tiles_per_batch * policy.onesweep.threads_per_block * policy.onesweep.items_per_thread;
  constexpr int capacity = 17 * batch_items + 37;
  const bool separate    = GENERATE(false, true);
  c2h::device_vector<uint32_t> first(capacity, thrust::no_init), second(capacity, thrust::no_init);
  c2h::gen(C2H_SEED(1), first);
  const c2h::host_vector<uint32_t> input = first;
  c2h::device_vector<uint64_t> device_count(1, uint64_t{0});
  cub::DoubleBuffer<uint32_t> keys(first.data().get(), second.data().get());
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  const auto count =
    cuda::args::deferred{device_count.data().get(), cuda::args::bounds<uint64_t{0}, uint64_t{capacity}>()};
  const auto env =
    cuda::std::execution::env{cuda::stream_ref{stream}, cuda::execution::tune(small_batch_policy_selector<>{})};
  size_t bytes{};
  auto sort = [&](void* scratch, size_t& size) {
    return separate ? cub::DeviceRadixSort::SortKeysDescending(
                        scratch, size, first.data().get(), second.data().get(), count, 0, 24, env)
                    : cub::DeviceRadixSort::SortKeysDescending(scratch, size, keys, count, 0, 24, env);
  };
  REQUIRE(cudaSuccess == sort(nullptr, bytes));
  c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
  cudaGraph_t graph{};
  REQUIRE(cudaSuccess == cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  REQUIRE(cudaSuccess == sort(scratch.data().get(), bytes));
  REQUIRE(cudaSuccess == cudaStreamEndCapture(stream, &graph));
  REQUIRE(keys.selector == (separate ? 0 : 1));
  cudaGraphExec_t executable{};
  REQUIRE(cudaSuccess == cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  for (const uint64_t n :
       {uint64_t{capacity}, uint64_t{0}, uint64_t{1}, uint64_t{1543}, uint64_t{capacity}, uint64_t{0}})
  {
    CAPTURE(n);
    first = input;
    REQUIRE(cudaSuccess == cudaMemsetAsync(second.data().get(), 0x5a, capacity * sizeof(uint32_t), stream));
    device_count[0] = n;
    REQUIRE(cudaSuccess == cudaGraphLaunch(executable, stream));
    REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));
    if (separate)
    {
      REQUIRE(input == first);
    }
    const c2h::device_vector<uint32_t> reference_input(input.begin(), input.begin() + n);
    const auto expected                     = radix_sort_reference(reference_input, true, 0, 24);
    const c2h::host_vector<uint32_t> output = second;
    REQUIRE(std::equal(expected.begin(), expected.begin() + n, output.begin()));
    REQUIRE(std::all_of(output.begin() + n, output.end(), [](uint32_t v) {
      return v == 0x5a5a5a5au;
    }));
  }
  REQUIRE(cudaSuccess == cudaGraphExecDestroy(executable));
  REQUIRE(cudaSuccess == cudaGraphDestroy(graph));
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

struct deferred_radix_select_predicate
{
  _CCCL_HOST_DEVICE_API bool operator()(uint32_t v) const
  {
    return v % 3 == 0;
  }
};

CUB_TEST("deferred radix consumes a same-stream selection count", "[device][radix_sort][deferred]", CUB_SMALL)
{
  constexpr int n = 40'000;
  c2h::device_vector<uint32_t> input(n), first(n, thrust::no_init), second(n, thrust::no_init);
  c2h::gen(C2H_SEED(2), input);
  const c2h::host_vector<uint32_t> host_input = input;
  c2h::host_vector<uint32_t> expected;
  for (uint32_t value : host_input)
  {
    if (deferred_radix_select_predicate{}(value))
    {
      expected.push_back(value);
    }
  }
  const c2h::device_vector<uint32_t> reference_input = expected;
  expected                                           = radix_sort_reference(reference_input, false);
  c2h::device_vector<int> device_count(1, -1);
  cub::DoubleBuffer<uint32_t> keys(first.data().get(), second.data().get());
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  size_t select_bytes{}, sort_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      nullptr,
      select_bytes,
      input.data().get(),
      first.data().get(),
      device_count.data().get(),
      n,
      deferred_radix_select_predicate{},
      stream));
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            nullptr, sort_bytes, keys, cuda::args::deferred{device_count.data().get()}, 0, 32, stream));
  c2h::device_vector<uint8_t> scratch(std::max(select_bytes, sort_bytes));
  REQUIRE(
    cudaSuccess
    == cub::DeviceSelect::If(
      scratch.data().get(),
      select_bytes,
      input.data().get(),
      first.data().get(),
      device_count.data().get(),
      n,
      deferred_radix_select_predicate{},
      stream));
  REQUIRE(cudaSuccess
          == cub::DeviceRadixSort::SortKeys(
            scratch.data().get(), sort_bytes, keys, cuda::args::deferred{device_count.data().get()}, 0, 32, stream));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(stream));
  const auto& sorted = keys.selector ? second : first;
  const c2h::host_vector<uint32_t> output(sorted.begin(), sorted.begin() + expected.size());
  REQUIRE(output == expected);
  REQUIRE(cudaSuccess == cudaStreamDestroy(stream));
}

using deferred_radix_batch_sizes = c2h::enum_type_list<int, 1, 2, 3, 2048>;

CUB_TEST("deferred radix progresses across batch sizes",
         "[device][radix_sort][deferred]",
         CUB_SMALL,
         deferred_radix_batch_sizes)
{
  constexpr int batch_tiles = c2h::get<0, TestType>::value;
  constexpr auto policy     = small_batch_policy_selector<batch_tiles>{}(cuda::compute_capability{7, 5});
  constexpr int batch_items = batch_tiles * policy.onesweep.threads_per_block * policy.onesweep.items_per_thread;
  // Tiny windows exercise frequent reuse with small grids; the larger window spans several batches with more CTAs.
  constexpr int batches = batch_tiles <= 2 ? 31 : 3;
  constexpr int n       = batch_items * batches + 17;
  c2h::device_vector<uint32_t> first(n, thrust::no_init), second(n, thrust::no_init);
  c2h::gen(C2H_SEED(1), first);
  const auto expected = radix_sort_reference(first, false);
  c2h::device_vector<int64_t> device_count(1, n);
  cub::DoubleBuffer<uint32_t> keys(first.data().get(), second.data().get());
  const auto count = cuda::args::deferred{device_count.data().get()};
  const auto env   = cuda::execution::tune(small_batch_policy_selector<batch_tiles>{});
  size_t bytes{};
  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, 0, 32, env));
  c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
  REQUIRE(cudaSuccess == cub::DeviceRadixSort::SortKeys(scratch.data().get(), bytes, keys, count, 0, 32, env));
  const c2h::host_vector<uint32_t> output = keys.selector ? second : first;
  REQUIRE(output == expected);
}

CUB_TEST("deferred radix environment APIs accept counts", "[device][radix_sort][deferred][env]", CUB_SMALL)
{
  const bool descending = GENERATE(false, true);
  const bool pairs      = GENERATE(false, true);
  const bool separate   = GENERATE(false, true);
  c2h::device_vector<uint32_t> first{7, 3, 7, 1, 3, 7}, second(6);
  c2h::device_vector<uint32_t> values_first{0, 1, 2, 3, 4, 5}, values_second(6);
  c2h::device_vector<int> device_count(1, 6);
  cub::DoubleBuffer<uint32_t> keys(first.data().get(), second.data().get());
  cub::DoubleBuffer<uint32_t> values(values_first.data().get(), values_second.data().get());
  const auto count = cuda::args::deferred{device_count.data().get(), cuda::args::bounds<0, 6>()};
  auto env         = cuda::execution::tune(small_batch_policy_selector<>{});
  if (separate)
  {
    auto* keys_in    = first.data().get();
    auto* keys_out   = second.data().get();
    auto* values_in  = values_first.data().get();
    auto* values_out = values_second.data().get();
    if (pairs)
    {
      REQUIRE(
        cudaSuccess
        == (descending
              ? cub::DeviceRadixSort::SortPairsDescending(keys_in, keys_out, values_in, values_out, count, 0, 24, env)
              : cub::DeviceRadixSort::SortPairs(keys_in, keys_out, values_in, values_out, count, 0, 24, env)));
    }
    else
    {
      REQUIRE(cudaSuccess
              == (descending ? cub::DeviceRadixSort::SortKeysDescending(keys_in, keys_out, count, 0, 24, env)
                             : cub::DeviceRadixSort::SortKeys(keys_in, keys_out, count, 0, 24, env)));
    }
    REQUIRE(first == c2h::host_vector<uint32_t>{7, 3, 7, 1, 3, 7});
    REQUIRE(values_first == c2h::host_vector<uint32_t>{0, 1, 2, 3, 4, 5});
  }
  else if (pairs)
  {
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortPairsDescending(keys, values, count, 0, 24, env)
                           : cub::DeviceRadixSort::SortPairs(keys, values, count, 0, 24, env)));
  }
  else
  {
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortKeysDescending(keys, count, 0, 24, env)
                           : cub::DeviceRadixSort::SortKeys(keys, count, 0, 24, env)));
  }
  const c2h::host_vector<uint32_t> output = separate || keys.selector ? second : first;
  const c2h::host_vector<uint32_t> expected =
    descending ? c2h::host_vector<uint32_t>{7, 7, 7, 3, 3, 1} : c2h::host_vector<uint32_t>{1, 3, 3, 7, 7, 7};
  REQUIRE(output == expected);
  REQUIRE(keys.selector == (separate ? 0 : 1));
  if (pairs)
  {
    const c2h::host_vector<uint32_t> output_values = separate || values.selector ? values_second : values_first;
    const c2h::host_vector<uint32_t> expected_values =
      descending ? c2h::host_vector<uint32_t>{0, 2, 5, 1, 4, 3} : c2h::host_vector<uint32_t>{3, 1, 4, 0, 2, 5};
    REQUIRE(output_values == expected_values);
  }
}

struct deferred_radix_decomposer
{
  template <template <typename> class... Policies>
  _CCCL_HOST_DEVICE_API cuda::std::tuple<size_t&> operator()(c2h::custom_type_t<Policies...>& key) const
  {
    return {key.key};
  }
};

CUB_TEST("deferred radix supports decomposers", "[device][radix_sort][deferred]", CUB_SMALL)
{
  using key_t           = c2h::custom_type_t<c2h::equal_comparable_t>;
  const bool descending = GENERATE(false, true);
  const bool separate   = GENERATE(false, true);
  const bool pairs      = GENERATE(false, true);
  constexpr int n       = 12345;
  c2h::device_vector<key_t> first(n), second(n);
  c2h::gen(C2H_SEED(1), first);
  c2h::host_vector<key_t> expected = first;
  std::stable_sort(expected.begin(), expected.end(), [&](const key_t& a, const key_t& b) {
    return descending ? a.key > b.key : a.key < b.key;
  });
  c2h::device_vector<int> device_count(1, n);
  cub::DoubleBuffer<key_t> keys(first.data().get(), second.data().get());
  const auto count                       = cuda::args::deferred{device_count.data().get(), cuda::args::bounds<0, n>()};
  auto env                               = cuda::execution::tune(small_batch_policy_selector<>{});
  const c2h::host_vector<key_t> original = first;
  if (separate && pairs)
  {
    c2h::device_vector<uint32_t> values_in(n, thrust::no_init), values_out(n, thrust::no_init);
    thrust::sequence(c2h::device_policy, values_in.begin(), values_in.end());
    auto* keys_in         = first.data().get();
    auto* keys_out        = second.data().get();
    auto* pairs_in        = values_in.data().get();
    auto* pairs_out       = values_out.data().get();
    const auto decomposer = deferred_radix_decomposer{};
    constexpr int end_bit = sizeof(size_t) * 8;
    // Exercise the explicit-storage decomposer overload with an explicit bit range.
    auto sort = [&](void* scratch, size_t& bytes) {
      return descending
             ? cub::DeviceRadixSort::SortPairsDescending(
                 scratch, bytes, keys_in, keys_out, pairs_in, pairs_out, count, decomposer, 0, end_bit, env)
             : cub::DeviceRadixSort::SortPairs(
                 scratch, bytes, keys_in, keys_out, pairs_in, pairs_out, count, decomposer, 0, end_bit, env);
    };
    size_t bytes{};
    REQUIRE(cudaSuccess == sort(nullptr, bytes));
    c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
    REQUIRE(cudaSuccess == sort(scratch.data().get(), bytes));
    c2h::host_vector<uint32_t> expected_values = values_in;
    std::stable_sort(expected_values.begin(), expected_values.end(), [&](uint32_t lhs, uint32_t rhs) {
      return descending ? original[lhs].key > original[rhs].key : original[lhs].key < original[rhs].key;
    });
    REQUIRE(expected_values == values_out);
  }
  else if (separate)
  {
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortKeysDescending(
                               first.data().get(), second.data().get(), count, deferred_radix_decomposer{}, env)
                           : cub::DeviceRadixSort::SortKeys(
                               first.data().get(), second.data().get(), count, deferred_radix_decomposer{}, env)));
  }
  else
  {
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortKeysDescending(keys, count, deferred_radix_decomposer{}, env)
                           : cub::DeviceRadixSort::SortKeys(keys, count, deferred_radix_decomposer{}, env)));
  }
  if (separate)
  {
    REQUIRE(original == first);
  }
  const c2h::host_vector<key_t> output = separate || keys.selector ? second : first;
  REQUIRE(output == expected);
}

CUB_TEST("deferred radix rejects invalid bit ranges", "[device][radix_sort][deferred]", CUB_SMALL)
{
  cub::DoubleBuffer<uint32_t> keys;
  const auto count = cuda::args::deferred{static_cast<int*>(nullptr)};
  size_t bytes{};
  REQUIRE(cudaErrorInvalidValue == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, -1, 32));
  REQUIRE(cudaErrorInvalidValue == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, 8, 7));
  REQUIRE(cudaErrorInvalidValue == cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count, 0, 33));
}

CUB_TEST("deferred radix scatters beyond 32-bit output offsets",
         "[device][radix_sort][deferred][large_offset][large][skip-cs-synccheck][skip-cs-initcheck][skip-cs-racecheck]",
         CUB_LARGE)
{
  using key_t           = uint8_t;
  constexpr uint64_t n  = (uint64_t{1} << 32) + 17;
  const bool descending = GENERATE(false, true);
  try
  {
    large_array_sort_helper<key_t> arrays;
    arrays.initialize_for_unstable_key_sort(C2H_SEED(1), n, descending);
    c2h::device_vector<uint64_t> device_count(1, n);
    const auto count = cuda::args::deferred{device_count.data().get()};
    auto& keys       = arrays.keys_buffer;
    size_t bytes{};
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortKeysDescending(nullptr, bytes, keys, count)
                           : cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, count)));
    c2h::device_vector<uint8_t> scratch(bytes, thrust::no_init);
    REQUIRE(cudaSuccess
            == (descending ? cub::DeviceRadixSort::SortKeysDescending(scratch.data().get(), bytes, keys, count)
                           : cub::DeviceRadixSort::SortKeys(scratch.data().get(), bytes, keys, count)));
    const auto& output = keys.selector ? arrays.keys_out : arrays.keys_in;
    arrays.verify_unstable_key_sort(n, descending, output);
  }
  catch (...)
  {
    c2h::skip_if_allocation_failure("Skipping deferred radix large-offset test: insufficient device memory");
  }
}
