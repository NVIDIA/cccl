// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/agent/agent_thread_segmented_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <thrust/tabulate.h>

#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "catch2_test_device_scan.cuh"
#include <c2h/catch2_test_helper.h>

namespace impl
{
template <typename UnsignedIntegralT>
using pair_t = cuda::std::pair<UnsignedIntegralT, UnsignedIntegralT>;

// bicyclid monoid operator is associative and non-commutative
template <typename UnsignedIntegralT>
struct bicyclic_monoid_op
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = pair_t<UnsignedIntegralT>;
  using min_t  = cuda::minimum<>;

  pair_t __host__ __device__ operator()(pair_t v1, pair_t v2)
  {
    auto [m, n] = v1;
    auto [r, s] = v2;
    auto min_nr = min_t{}(n, r);
    return {m + r - min_nr, s + n - min_nr};
  }
};

template <typename UnsignedIntegralT>
struct populate_input
{
  static_assert(cuda::std::is_integral_v<UnsignedIntegralT>);
  static_assert(cuda::std::is_unsigned_v<UnsignedIntegralT>);

  using pair_t = pair_t<UnsignedIntegralT>;

  __host__ __device__ pair_t operator()(size_t id) const
  {
    static constexpr pair_t short_seq[] = {
      {0, 1}, {2, 3}, {4, 1}, {2, 5}, {7, 1}, {1, 1}, {0, 4}, {3, 1}, {1, 2}, {3, 2}, {4, 5}, {3, 5},
      {1, 9}, {0, 1}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {1, 0}, {2, 2}, {2, 2}, {0, 0}, {1, 1}, {2, 3},
      {2, 4}, {4, 3}, {1, 3}, {0, 3}, {1, 1}, {5, 1}, {2, 3}, {4, 7}, {2, 6}, {8, 3}, {1, 0}, {0, 8}};

    static constexpr size_t nelems = sizeof(short_seq) / sizeof(pair_t);

    return short_seq[id % nelems];
  }
};
}; // namespace impl

namespace
{
template <int BlockThreads, int ItemsPerThread, cub::CacheLoadModifier LoadModifier>
struct agent_policy_t
{
  static constexpr int BLOCK_THREADS                    = BlockThreads;
  static constexpr int ITEMS_PER_THREAD                 = ItemsPerThread;
  static constexpr cub::CacheLoadModifier load_modifier = LoadModifier;
};

template <int BlockThreads, int ItemsPerThread>
struct policy_wrapper
{
  using thread_segmented_scan_policy_t = agent_policy_t<BlockThreads, ItemsPerThread, cub::LOAD_DEFAULT>;
};

template <int BlockThreads, int ItemsPerThread>
struct ChainedPolicy
{
  using ActivePolicy = policy_wrapper<BlockThreads, ItemsPerThread>;
};
} // namespace

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with one segment per block",
         "[agent_single_segment_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 16;
  c2h::device_vector<unsigned> offsets{0u, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  constexpr int block_size       = 128;
  constexpr int items_per_thread = 4;
  using chained_policy_t         = ChainedPolicy<block_size, items_per_thread>;

  static constexpr int segments_per_worker = 1;
  static constexpr int segments_per_block  = segments_per_worker * block_size;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  [[maybe_unused]] const auto itp = items_per_thread;

  op_t scan_op{};
  cub::NullType no_init{};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_offsets;

  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    true><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, no_init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      scan_op,
      pair_t{0, 0});
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with two segments per block",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 14 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  constexpr int block_size       = 128;
  constexpr int items_per_thread = 4;
  using chained_policy_t         = ChainedPolicy<block_size, items_per_thread>;

  constexpr int segments_per_worker = 2;
  const int segments_per_block      = segments_per_worker * block_size;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  assert(grid_size * segments_per_block == n_segments);

  [[maybe_unused]] const auto itp = items_per_thread;

  op_t scan_op{};
  cub::NullType no_init{};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_offsets;

  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    false><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, no_init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      scan_op,
      pair_t{0, 0});
  }

  if (h_expected != h_output)
  {
    for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
    {
      for (std::size_t i = h_offsets[segment_id]; i < h_offsets[segment_id + 1]; ++i)
      {
        std::cout << "({" << h_output[i].first << ", " << h_output[i].second << "}, {" << h_expected[i].first << ", "
                  << h_expected[i].second << "}) ";
      }
      std::cout << std::endl;
    }
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan works with three segments per block and tail",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 128 * 17 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  constexpr int block_size       = 128;
  constexpr int items_per_thread = 4;
  using chained_policy_t         = ChainedPolicy<block_size, items_per_thread>;

  static constexpr int segments_per_worker = 3;
  static constexpr int segments_per_block  = segments_per_worker * block_size;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  [[maybe_unused]] const auto itp = items_per_thread;

  op_t scan_op{};
  cub::NullType no_init{};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_offsets;

  // inclusive scan (no initial condition)
  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    pair_t,
    false><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, no_init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      scan_op,
      pair_t{0, 0});
  }

  if (h_expected != h_output)
  {
    for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
    {
      for (std::size_t i = h_offsets[segment_id]; i < h_offsets[segment_id + 1]; ++i)
      {
        std::cout << "({" << h_output[i].first << ", " << h_output[i].second << "}, {" << h_expected[i].first << ", "
                  << h_expected[i].second << "}) ";
      }
      std::cout << std::endl;
    }
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("cub::detail::segmented_scan::agent_segmented_scan skips empty segments",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t    = cuda::std::plus<>;
  using value_t = unsigned;

  auto canary = value_t{0xDEADBEEF};

  c2h::device_vector<unsigned> offsets{0, 4, 4, 12};
  c2h::device_vector<unsigned> out_offsets{0, 4, 8, 16};

  size_t num_segments = offsets.size() - 1;
  unsigned num_items  = offsets.back();
  unsigned num_output = out_offsets.back();

  c2h::device_vector<value_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), cuda::std::identity{});
  c2h::device_vector<value_t> output(num_output, canary);

  value_t* d_input        = thrust::raw_pointer_cast(input.data());
  value_t* d_output       = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets     = thrust::raw_pointer_cast(offsets.data());
  unsigned* d_out_offsets = thrust::raw_pointer_cast(out_offsets.data());

  constexpr int block_size          = 128;
  constexpr int items_per_thread    = 4;
  constexpr int segments_per_worker = 2;
  constexpr int segments_per_block  = segments_per_worker * block_size;
  using chained_policy_t            = ChainedPolicy<block_size, items_per_thread>;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  [[maybe_unused]] const auto itp = items_per_thread;

  op_t scan_op{};
  cub::NullType no_init{};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_out_offsets;

  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    value_t*,
    value_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::NullType,
    value_t,
    false><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, no_init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  c2h::host_vector<value_t> h_output(output);
  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<value_t> h_expected(output.size(), canary);
  c2h::host_vector<unsigned> h_offsets(offsets);
  c2h::host_vector<unsigned> h_out_offsets(out_offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    if (h_offsets[segment_id] == h_offsets[segment_id + 1])
    {
      continue;
    }
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_out_offsets[segment_id],
      scan_op,
      value_t{0});
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("agent_segmented_scan works for exclusive_scan with two segments per block and tail",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t   = impl::bicyclic_monoid_op<unsigned>;
  using pair_t = typename op_t::pair_t;

  unsigned num_items = 3 * 1 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<pair_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), impl::populate_input<unsigned>{});
  c2h::device_vector<pair_t> output(input.size());

  pair_t* d_input     = thrust::raw_pointer_cast(input.data());
  pair_t* d_output    = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  constexpr int block_size          = 128;
  constexpr int items_per_thread    = 4;
  constexpr int segments_per_worker = 2;
  constexpr int segments_per_block  = segments_per_worker * block_size;
  using chained_policy_t            = ChainedPolicy<block_size, items_per_thread>;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  [[maybe_unused]] const auto itp = items_per_thread;

  pair_t h_init{1, 1};

  op_t scan_op{};
  cub::detail::InputValue<pair_t> init{h_init};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_offsets;

  // force inclusive is false (last template parameter), initial value is provided
  // hence this call computes exclusive scan algorithm
  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    pair_t*,
    pair_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::detail::InputValue<pair_t>,
    pair_t,
    false><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<pair_t> h_output(output);
  c2h::host_vector<pair_t> h_input(input);
  c2h::host_vector<pair_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_exclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      h_init,
      scan_op);
  }

  if (h_expected != h_output)
  {
    for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
    {
      for (std::size_t i = h_offsets[segment_id]; i < h_offsets[segment_id + 1]; ++i)
      {
        std::cout << "({" << h_output[i].first << ", " << h_output[i].second << "}, {" << h_expected[i].first << ", "
                  << h_expected[i].second << "}) ";
      }
      std::cout << std::endl;
    }
  }

  REQUIRE(h_expected == h_output);
}

C2H_TEST("agent_segmented_scan works for exclusive_scan with three segments per block and tail",
         "[agent_multiple_segments_per_block][segmented][scan]")
{
  using op_t    = cuda::std::plus<>;
  using value_t = unsigned;

  unsigned num_items = 3 * 1 * 4;
  c2h::device_vector<unsigned> offsets{0, num_items / 4, num_items / 2, num_items - (num_items / 4), num_items};
  size_t num_segments = offsets.size() - 1;

  c2h::device_vector<value_t> input(num_items);
  thrust::tabulate(input.begin(), input.end(), cuda::std::identity{});
  c2h::device_vector<value_t> output(input.size());

  value_t* d_input    = thrust::raw_pointer_cast(input.data());
  value_t* d_output   = thrust::raw_pointer_cast(output.data());
  unsigned* d_offsets = thrust::raw_pointer_cast(offsets.data());

  constexpr int block_size          = 128;
  constexpr int items_per_thread    = 4;
  constexpr int segments_per_worker = 3;
  constexpr int segments_per_block  = segments_per_worker * block_size;
  using chained_policy_t            = ChainedPolicy<block_size, items_per_thread>;

  const auto n_segments = static_cast<unsigned>(num_segments);
  const auto grid_size  = cuda::ceil_div(n_segments, segments_per_block);

  [[maybe_unused]] const auto itp = items_per_thread;

  value_t h_init{10};

  op_t scan_op{};
  cub::detail::InputValue<value_t> init{h_init};
  auto inp_beg = d_offsets;
  auto inp_end = d_offsets + 1;
  auto out_beg = d_offsets;

  // force inclusive is false (last template parameter), initial value is provided
  // hence this call computes inclusive scan algorithm
  cub::detail::segmented_scan::device_thread_segmented_scan_kernel<
    chained_policy_t,
    value_t*,
    value_t*,
    unsigned*,
    unsigned*,
    unsigned*,
    unsigned,
    op_t,
    cub::detail::InputValue<value_t>,
    value_t,
    true><<<grid_size, block_size>>>(
    d_input, d_output, inp_beg, inp_end, out_beg, n_segments, scan_op, init, segments_per_worker);

  REQUIRE(cudaSuccess == cudaGetLastError());

  // transfer to host_vector is synchronizing
  c2h::host_vector<value_t> h_output(output);
  c2h::host_vector<value_t> h_input(input);
  c2h::host_vector<value_t> h_expected(input.size());
  c2h::host_vector<unsigned> h_offsets(offsets);

  for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
  {
    compute_inclusive_scan_reference(
      h_input.begin() + h_offsets[segment_id],
      h_input.begin() + h_offsets[segment_id + 1],
      h_expected.begin() + h_offsets[segment_id],
      scan_op,
      h_init);
  }

  if (h_expected != h_output)
  {
    for (unsigned segment_id = 0; segment_id < num_segments; ++segment_id)
    {
      for (std::size_t i = h_offsets[segment_id]; i < h_offsets[segment_id + 1]; ++i)
      {
        std::cout << "(" << h_output[i] << ", " << h_expected[i] << ") ";
      }
      std::cout << std::endl;
    }
  }

  REQUIRE(h_expected == h_output);
}
