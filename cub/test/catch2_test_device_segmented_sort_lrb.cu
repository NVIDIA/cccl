// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Focused checks for capacity mapping, ID partitioning, and the internal segmented sorting path.

#include <cub/device/dispatch/dispatch_segmented_sort_lrb.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort_group_merge.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>
#include <c2h/device_and_stream.h>

namespace capacity    = cub::detail::segmented_sort_lrb;
constexpr int sm90_cc = cuda::compute_capability{9, 0}.get();
using sort_orders     = c2h::enum_type_list<cub::SortOrder, cub::SortOrder::Ascending, cub::SortOrder::Descending>;

// Exercise host/device projections with architecture-dependent private settings.
template <typename Key>
struct test_policy_selector
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const
  {
    auto policy                                   = capacity::policy_selector<Key>{}(cc);
    policy.partition.threads_per_block            = cc >= cuda::compute_capability{10, 0} ? 128 : 256;
    policy.group_merge.warp.min_items_per_thread  = cc >= cuda::compute_capability{10, 0} ? 10 : 9;
    policy.group_merge.large.grid_size_multiplier = 1;
    return policy;
  }
};

// Irregular capacities exercise bin selection independently of rectangular tuning.
struct irregular_expansion
{
  using capacities = capacity::capacity_list<36, 80, 144>;
  template <int Bin>
  using shape_for_bin = cuda::std::tuple_element_t<
    Bin,
    cuda::std::tuple<capacity::bin_shape<4, 9>, capacity::bin_shape<8, 10>, capacity::bin_shape<16, 9>>>;
};

// Check generic bounds, irregular configuration mapping, overflow IDs, and empty-workspace reuse.
CUB_TEST("Capacity partition accepts arbitrary bounds without sort tuning", "[segmented][sort][device]", CUB_SMALL)
{
  for (int bin_idx = 0; bin_idx < 3; ++bin_idx)
  {
    const auto config = capacity::lookup_bin_config<irregular_expansion>(bin_idx);
    REQUIRE(config.threads_per_segment == (4 << bin_idx));
    REQUIRE(config.items_per_thread == (bin_idx == 1 ? 10 : 9));
  }
  using policies     = test_policy_selector<int>;
  using kernels      = capacity::group_merge_sort_kernels_t<int, policies>;
  using sm90_policy  = cub::detail::policy_getter<policies, cuda::compute_capability{9, 0}.get()>;
  using sm100_policy = cub::detail::policy_getter<policies, cuda::compute_capability{10, 0}.get()>;
  static_assert(sm90_policy{}().partition.threads_per_block == 256);
  static_assert(sm100_policy{}().partition.threads_per_block == 128);
  static_assert(kernels::capacities_t<sm90_policy>::size == 77);
  static_assert(kernels::capacities_t<sm100_policy>::size == 71);

  using capacities = capacity::capacity_list<3, 10, 21>;
  static_assert(capacities::first_fit(3) == 0 && capacities::first_fit(4) == 1);
  static_assert(capacities::first_fit(21) == 2 && capacities::first_fit(22) == 3);
  const c2h::host_vector<int> begins{5, 7, 9, 20, 30, 40, 50, 60, 90};
  const c2h::host_vector<int> ends{4, 7, 10, 23, 34, 50, 61, 81, 112};
  c2h::device_vector<int> d_begins = begins;
  c2h::device_vector<int> d_ends   = ends;
  capacity::partition_result result;
  size_t bytes   = 0;
  const auto run = [&](void* storage, unsigned count) {
    return capacity::dispatch_partition<capacities>(
      storage, bytes, thrust::raw_pointer_cast(d_begins.data()), thrust::raw_pointer_cast(d_ends.data()), count, result);
  };
  REQUIRE(run(nullptr, static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), static_cast<unsigned>(begins.size())) == cudaSuccess);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>{0, 2, 4, 6, 7});
  c2h::host_vector<unsigned> ids(offsets.back());
  REQUIRE(
    cudaMemcpy(ids.data(), result.segment_ids, ids.size() * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
  for (int bin_idx = 0; bin_idx <= capacities::size; ++bin_idx)
  {
    for (unsigned i = offsets[bin_idx]; i < offsets[bin_idx + 1]; ++i)
    {
      REQUIRE(capacities::first_fit(ends[ids[i]] - begins[ids[i]]) == bin_idx);
    }
  }
  std::sort(ids.begin(), ids.end());
  REQUIRE(ids == c2h::host_vector<unsigned>{2, 3, 4, 5, 6, 7, 8});
  // Reuse the same workspace with no segments; every range must become empty.
  REQUIRE(run(thrust::raw_pointer_cast(storage.data()), 0) == cudaSuccess);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>(capacities::size + 2, 0));
}

// Retry one sorting kernel with a reduced grid and check that it restores exactly its segment ranges.
template <int KernelIndex, cub::SortOrder Order, typename Key>
static void check_kernel_retry(
  capacity::partition_result bins,
  const c2h::host_vector<long long>& begins,
  const c2h::host_vector<long long>& ends,
  const c2h::host_vector<Key>& expected,
  long long* d_begins,
  long long* d_ends,
  const Key* d_in,
  c2h::device_vector<Key>& d_out,
  cudaStream_t stream)
{
  using kernels_t           = capacity::group_merge_sort_kernels_t<Key>;
  using sort_policy_t       = cub::detail::policy_getter<capacity::policy_selector<Key>, sm90_cc>;
  using capacities          = typename kernels_t::template capacities_t<sort_policy_t>;
  using kernel_t            = typename kernels_t::template kernel_t<KernelIndex>;
  using policy_getter_t     = cub::detail::policy_getter<typename kernel_t::policy_selector, sm90_cc>;
  using kernel_capacities_t = typename kernel_t::template capacities_t<policy_getter_t>;
  constexpr int first_bin   = kernels_t::template kernel_bin_begin_idx<sort_policy_t>(KernelIndex);
  constexpr int lower_bound = first_bin ? capacities::bounds[static_cast<size_t>(first_bin - 1)] : 0;
  auto retry                = expected;
  for (size_t segment = 0; segment < begins.size(); ++segment)
  {
    const auto length = ends[segment] - begins[segment];
    if (length > lower_bound && length <= kernel_capacities_t::bounds.back())
    {
      std::fill(retry.begin() + begins[segment], retry.begin() + ends[segment], Key{-99});
    }
  }
  d_out = retry;
  REQUIRE(
    capacity::launch_sort_kernel<kernel_t, Order>(
      {bins.bin_offsets + first_bin, bins.segment_ids},
      d_begins,
      d_ends,
      d_in,
      thrust::raw_pointer_cast(d_out.data()),
      1,
      stream,
      cub::detail::TripleChevronFactory{},
      policy_getter_t{})
    == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<Key>(d_out) == expected);
}

template <cub::SortOrder Order, typename... Args>
static void check_kernel_retries(Args&&... args)
{
  check_kernel_retry<0, Order>(args...);
  check_kernel_retry<1, Order>(args...);
  check_kernel_retry<2, Order>(args...);
}

// Exercise capacity boundaries and all launch groups, including repeated work and radix overflow.
CUB_TEST("Capacity sort reconstructs work prefixes in each CTA",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<int, double>,
         sort_orders)
{
  using key_type   = c2h::get<0, TestType>;
  using order_t    = c2h::get<1, TestType>;
  using compare_op = std::conditional_t<order_t::value == cub::SortOrder::Ascending, std::less<>, std::greater<>>;
  using kernels_t  = capacity::group_merge_sort_kernels_t<key_type>;
  using capacities =
    typename kernels_t::template capacities_t<cub::detail::policy_getter<capacity::policy_selector<key_type>, sm90_cc>>;
  using warp_kernel_t        = typename kernels_t::template kernel_t<0>;
  using warp_policy_getter_t = cub::detail::policy_getter<typename warp_kernel_t::policy_selector, sm90_cc>;
  using warp_expansion       = typename warp_kernel_t::template expansion_t<warp_policy_getter_t>;
  const int bin_idx          = warp_expansion::capacities::first_fit(37);
  const auto config          = capacity::lookup_bin_config<warp_expansion>(bin_idx);
  REQUIRE(config.threads_per_segment == 4);
  REQUIRE(config.items_per_thread == 10);
  std::vector<int> lengths{0, 1};
  for (const int bound : capacities::bounds)
  {
    for (const int delta : {-1, 0, 1})
    {
      lengths.push_back(bound + delta);
    }
  }
  // A partially filled final group, many work iterations, and absent bins.
  lengths.insert(lengths.end(), 4097, warp_expansion::capacities::bounds.back());
  using small_kernel_t        = typename kernels_t::template kernel_t<1>;
  using small_policy_getter_t = cub::detail::policy_getter<typename small_kernel_t::policy_selector, sm90_cc>;
  using small_capacities      = typename small_kernel_t::template capacities_t<small_policy_getter_t>;
  lengths.insert(lengths.end(), 1000, small_capacities::bounds.back());
  lengths.insert(lengths.end(), 7, capacities::bounds.back() - 1);
  c2h::host_vector<long long> begins;
  c2h::host_vector<long long> ends;
  c2h::host_vector<key_type> input;
  c2h::host_vector<key_type> expected;
  std::mt19937 random{12345};
  std::uniform_int_distribution<int> random_key{-100000, 100000};
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    const int length = lengths[segment];
    // Gaps also verify that only each requested range is written.
    input.push_back(key_type{-99});
    begins.push_back(input.size());
    for (int i = 0; i < length; ++i)
    {
      // Alternate random, duplicate-heavy, and descending inputs, including negative doubles.
      const int value = segment % 3 == 0 ? random_key(random)
                      : segment % 3 == 1 ? random_key(random) % 7
                                         : ((length - i) % 257) - 128;
      input.push_back(static_cast<key_type>(value) / key_type{4});
    }
    ends.push_back(input.size());
  }
  expected.assign(input.size(), key_type{-99});
  for (size_t segment = 0; segment < lengths.size(); ++segment)
  {
    std::copy(input.begin() + begins[segment], input.begin() + ends[segment], expected.begin() + begins[segment]);
    std::stable_sort(expected.begin() + begins[segment], expected.begin() + ends[segment], compare_op{});
  }
  c2h::device_vector<key_type> d_in = input;
  c2h::device_vector<key_type> d_out(input.size(), key_type{-99});
  c2h::device_vector<long long> d_begins = begins;
  c2h::device_vector<long long> d_ends   = ends;
  capacity::partition_result result;
  unsigned num_segments     = static_cast<unsigned>(lengths.size());
  size_t bytes              = 0;
  const auto stream_owner   = c2h::make_current_device_stream();
  const cudaStream_t stream = stream_owner.get();
  const auto run            = [&](void* storage) {
    return capacity::dispatch<order_t::value>(
      storage,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      num_segments,
      static_cast<cuda::std::int64_t>(input.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result,
      stream);
  };
  const auto query_status = run(nullptr);
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc < cuda::compute_capability{9, 0})
  {
    REQUIRE(query_status == cudaErrorNotSupported);
    return;
  }
  REQUIRE(query_status == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  c2h::host_vector<unsigned> offsets(capacities::size + 2);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets.back() - offsets[capacities::size] == 1);
  REQUIRE(c2h::host_vector<key_type>(d_in) == input);
  // Force scratch reuse and cross-bin work iterations independently in every sorting kernel.
  check_kernel_retries<order_t::value>(
    result,
    begins,
    ends,
    expected,
    thrust::raw_pointer_cast(d_begins.data()),
    thrust::raw_pointer_cast(d_ends.data()),
    thrust::raw_pointer_cast(d_in.data()),
    d_out,
    stream);
  // A different private aggregate must drive matching partition and sorting settings on both sides.
  thrust::fill(d_out.begin(), d_out.end(), key_type{-99});
  const auto run_custom = [&](void* workspace) {
    return capacity::dispatch<order_t::value,
                              key_type,
                              long long*,
                              long long*,
                              cub::detail::TripleChevronFactory,
                              test_policy_selector<key_type>>(
      workspace,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<unsigned>(lengths.size()),
      static_cast<cuda::std::int64_t>(input.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result,
      stream);
  };
  REQUIRE(run_custom(nullptr) == cudaSuccess);
  storage.resize(bytes);
  REQUIRE(run_custom(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);
  // Restore the default workspace size before testing default-policy reuse.
  REQUIRE(run(nullptr) == cudaSuccess);
  storage.resize(bytes);
  // Prefixes and scatter cursors must reset on a subsequent invocation.
  thrust::fill(d_out.begin(), d_out.end(), key_type{-99});
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(c2h::host_vector<key_type>(d_out) == expected);

  // Sort the same mix of bounded and overflow segments starting in buffer 1.
  cub::DoubleBuffer<key_type> keys(thrust::raw_pointer_cast(d_out.data()), thrust::raw_pointer_cast(d_in.data()));
  keys.selector          = 1;
  size_t buffer_bytes    = 0;
  const auto run_buffers = [&](void* workspace) {
    return capacity::dispatch<order_t::value>(
      workspace,
      buffer_bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<unsigned>(lengths.size()),
      static_cast<cuda::std::int64_t>(input.size()),
      keys,
      true,
      result,
      stream);
  };
  REQUIRE(run_buffers(nullptr) == cudaSuccess);
  REQUIRE(keys.selector == 1);
  REQUIRE(buffer_bytes < bytes);
  c2h::device_vector<unsigned char> buffer_storage(buffer_bytes);
  thrust::fill(d_out.begin(), d_out.end(), key_type{-99});
  for (int iteration = 0; iteration < 2; ++iteration)
  {
    REQUIRE(run_buffers(thrust::raw_pointer_cast(buffer_storage.data())) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(c2h::host_vector<key_type>(keys.selector ? d_in : d_out) == expected);
  }

  num_segments = 0;
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  REQUIRE(cudaMemcpy(offsets.data(), result.bin_offsets, offsets.size() * sizeof(unsigned), cudaMemcpyDeviceToHost)
          == cudaSuccess);
  REQUIRE(offsets == c2h::host_vector<unsigned>(capacities::size + 2, 0));
}

CUB_TEST("LRB partition storage is capped at one window", "[segmented][sort][device]", CUB_SMALL)
{
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (!capacity::can_use_lrb<int>(cc))
  {
    SKIP("LRB requires a compiled SM90-or-newer target.");
  }

  capacity::partition_result result;
  size_t bytes     = 0;
  int* ptr         = nullptr;
  const auto query = [&](std::int64_t num_segments) {
    return capacity::dispatch<cub::SortOrder::Ascending>(nullptr, bytes, ptr, ptr, num_segments, 1, ptr, ptr, result);
  };
  constexpr std::int64_t int_max  = std::numeric_limits<int>::max();
  constexpr std::int64_t uint_max = std::numeric_limits<unsigned>::max();
  REQUIRE(query(-1) == cudaErrorInvalidValue);
  REQUIRE(query(int_max - 1) == cudaSuccess);
  const auto below_limit = bytes;
  REQUIRE(query(int_max) == cudaSuccess);
  const auto at_limit = bytes;
  REQUIRE(at_limit >= below_limit);
  REQUIRE(at_limit >= sizeof(unsigned) * static_cast<size_t>(int_max));
  for (const auto count : {int_max + 1, uint_max - 1, uint_max, uint_max + 1, std::numeric_limits<std::int64_t>::max()})
  {
    CAPTURE(count);
    REQUIRE(query(count) == cudaSuccess);
    REQUIRE(bytes == at_limit);
  }
}

CUB_TEST("LRB reuses partition storage across segment windows",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<std::uint16_t, std::uint32_t>,
         c2h::enum_type_list<cub::SortOrder, cub::SortOrder::Ascending, cub::SortOrder::Descending>)
{
  using key_t          = c2h::get<0, TestType>;
  constexpr auto order = c2h::get<1, TestType>::value;
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (!capacity::can_use_lrb<key_t>(cc))
  {
    SKIP("LRB requires a compiled SM90-or-newer target.");
  }

  constexpr int window_size = 4;
  // Two mixed windows, an empty window, and a partial window with both sort paths.
  const std::vector<int> lengths{0, 16001, 2, 0, 3, 0, 16000, 1, 0, 0, 0, 0, 5, 16004};
  c2h::host_vector<int> begins, ends;
  c2h::host_vector<key_t> input;
  for (const auto length : lengths)
  {
    input.push_back(17); // Gap between segments.
    begins.push_back(static_cast<int>(input.size()));
    for (int i = 0; i < length; ++i)
    {
      input.push_back(static_cast<key_t>((i * 73) % 997));
    }
    ends.push_back(static_cast<int>(input.size()));
  }
  auto expected = input;
  for (size_t i = 0; i < lengths.size(); ++i)
  {
    std::sort(expected.begin() + begins[i], expected.begin() + ends[i], [](key_t lhs, key_t rhs) {
      return order == cub::SortOrder::Ascending ? lhs < rhs : lhs > rhs;
    });
  }

  const bool overwrite              = GENERATE(false, true);
  const int initial_selector        = GENERATE(0, 1);
  c2h::device_vector<int> d_begins  = begins;
  c2h::device_vector<int> d_ends    = ends;
  c2h::device_vector<key_t> d_input = input;
  c2h::device_vector<key_t> d_alternate(input.size(), key_t{17});
  auto* in        = thrust::raw_pointer_cast(d_input.data());
  auto* alternate = thrust::raw_pointer_cast(d_alternate.data());
  cub::DoubleBuffer<key_t> keys(initial_selector ? alternate : in, initial_selector ? in : alternate);
  keys.selector = initial_selector;
  capacity::partition_result result;
  size_t bytes            = 0;
  const auto stream_owner = c2h::make_current_device_stream();
  const auto run          = [&](void* storage) {
    return capacity::dispatch<order>(
      storage,
      bytes,
      thrust::raw_pointer_cast(d_begins.data()),
      thrust::raw_pointer_cast(d_ends.data()),
      static_cast<std::int64_t>(lengths.size()),
      static_cast<std::int64_t>(input.size()),
      keys,
      overwrite,
      result,
      stream_owner.get(),
      cub::detail::TripleChevronFactory{},
      window_size);
  };
  REQUIRE(run(nullptr) == cudaSuccess);
  REQUIRE(keys.selector == initial_selector);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream_owner.get()) == cudaSuccess);
  const auto& output = keys.Current() == in ? d_input : d_alternate;
  REQUIRE(c2h::host_vector<key_t>(output) == expected);
  if (!overwrite)
  {
    REQUIRE(keys.Current() == alternate);
    REQUIRE(c2h::host_vector<key_t>(d_input) == input);
  }

  // The final window has one bounded segment followed by one overflow segment.
  unsigned ids[2]{};
  REQUIRE(cudaMemcpy(ids, result.segment_ids, sizeof(ids), cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(ids[0] == 0);
  REQUIRE(ids[1] == 1);
}

// Check supported floating-point comparisons without imposing radix-sort signed-zero ordering.
CUB_TEST("Capacity sort orders infinities and treats signed zeros as equivalent",
         "[segmented][sort][device]",
         CUB_SMALL,
         c2h::type_list<float, double>,
         sort_orders)
{
  using key_type          = c2h::get<0, TestType>;
  using order_t           = c2h::get<1, TestType>;
  const key_type infinity = std::numeric_limits<key_type>::infinity();
  const c2h::host_vector<key_type> input{infinity, -key_type{0}, 3, -infinity, key_type{0}, -7, -1, 1};
  const c2h::host_vector<key_type> expected =
    order_t::value == cub::SortOrder::Ascending
      ? c2h::host_vector<key_type>{-infinity, -7, -1, 0, 0, 1, 3, infinity}
      : c2h::host_vector<key_type>{infinity, 3, 1, 0, 0, -1, -7, -infinity};
  c2h::device_vector<key_type> d_in = input;
  c2h::device_vector<key_type> d_out(input.size());
  c2h::device_vector<int> d_offsets = c2h::host_vector<int>{0, static_cast<int>(input.size())};
  const auto stream_owner           = c2h::make_current_device_stream();
  const cudaStream_t stream         = stream_owner.get();
  capacity::partition_result result;
  size_t bytes   = 0;
  const auto run = [&](void* storage) {
    const auto* offsets = thrust::raw_pointer_cast(d_offsets.data());
    return capacity::dispatch<order_t::value>(
      storage,
      bytes,
      offsets,
      offsets + 1,
      1,
      static_cast<cuda::std::int64_t>(input.size()),
      thrust::raw_pointer_cast(d_in.data()),
      thrust::raw_pointer_cast(d_out.data()),
      result,
      stream);
  };
  const auto query_status = run(nullptr);
  cuda::compute_capability cc{};
  REQUIRE(cub::detail::ptx_compute_cap(cc) == cudaSuccess);
  if (cc < cuda::compute_capability{9, 0})
  {
    REQUIRE(query_status == cudaErrorNotSupported);
    return;
  }
  REQUIRE(query_status == cudaSuccess);
  c2h::device_vector<unsigned char> storage(bytes);
  REQUIRE(run(thrust::raw_pointer_cast(storage.data())) == cudaSuccess);
  REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
  const c2h::host_vector<key_type> output = d_out;
  REQUIRE(output == expected);
  // Both zero signs are preserved and adjacent; their relative order is unspecified.
  REQUIRE(std::signbit(output[3]) != std::signbit(output[4]));
}
