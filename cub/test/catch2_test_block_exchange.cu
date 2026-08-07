// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/block/block_exchange.cuh>

#include <thrust/memory.h>

#include <cstdint>

#include <c2h/catch2_test_helper.h>

enum class block_exchange_op
{
  striped_to_blocked,
  blocked_to_striped,
  warp_striped_to_blocked,
  blocked_to_warp_striped,
  scatter_to_blocked,
  scatter_to_striped,
  scatter_to_striped_guarded,
  scatter_to_striped_flagged
};

template <block_exchange_op Op, int BlockThreads, int ItemsPerThread, typename T>
__global__ void block_exchange_kernel(T* data)
{
  using block_exchange_t = cub::BlockExchange<T, BlockThreads, ItemsPerThread>;
  using temp_storage_t   = typename block_exchange_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

  constexpr int tile_size = BlockThreads * ItemsPerThread;
  const int tid           = static_cast<int>(threadIdx.x);
  const int thread_offset = tid * ItemsPerThread;

  T items[ItemsPerThread];
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    items[item] = data[thread_offset + item];
  }

  block_exchange_t exchange(temp_storage);

  if constexpr (Op == block_exchange_op::striped_to_blocked)
  {
    exchange.StripedToBlocked(items);
  }
  else if constexpr (Op == block_exchange_op::blocked_to_striped)
  {
    exchange.BlockedToStriped(items);
  }
  else if constexpr (Op == block_exchange_op::warp_striped_to_blocked)
  {
    exchange.WarpStripedToBlocked(items);
  }
  else if constexpr (Op == block_exchange_op::blocked_to_warp_striped)
  {
    exchange.BlockedToWarpStriped(items);
  }
  else
  {
    int ranks[ItemsPerThread];
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int item_offset = thread_offset + item;
      ranks[item]           = tile_size - 1 - item_offset;
    }

    if constexpr (Op == block_exchange_op::scatter_to_blocked)
    {
      exchange.ScatterToBlocked(items, ranks);
    }
    else if constexpr (Op == block_exchange_op::scatter_to_striped)
    {
      exchange.ScatterToStriped(items, ranks);
    }
    else if constexpr (Op == block_exchange_op::scatter_to_striped_guarded)
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        if ((thread_offset + item) % 2 != 0)
        {
          ranks[item] = -1;
        }
      }
      exchange.ScatterToStripedGuarded(items, ranks);
    }
    else
    {
      bool is_valid[ItemsPerThread];
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        is_valid[item] = (thread_offset + item) % 2 == 0;
      }
      exchange.ScatterToStripedFlagged(items, ranks, is_valid);
    }
  }

  for (int item = 0; item < ItemsPerThread; ++item)
  {
    data[thread_offset + item] = items[item];
  }
}

template <block_exchange_op Op, int BlockThreads, int ItemsPerThread, typename T>
void invoke_block_exchange(c2h::device_vector<T>& data)
{
  block_exchange_kernel<Op, BlockThreads, ItemsPerThread><<<1, BlockThreads>>>(thrust::raw_pointer_cast(data.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename T>
c2h::host_vector<T>
blocked_to_striped_reference(const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  c2h::host_vector<T> output(input.size());
  for (int tid = 0; tid < block_threads; ++tid)
  {
    for (int item = 0; item < items_per_thread; ++item)
    {
      output[tid * items_per_thread + item] = input[item * block_threads + tid];
    }
  }
  return output;
}

template <typename T>
c2h::host_vector<T>
striped_to_blocked_reference(const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  c2h::host_vector<T> output(input.size());
  for (int tid = 0; tid < block_threads; ++tid)
  {
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int output_offset = tid * items_per_thread + item;
      const int input_tid     = output_offset % block_threads;
      const int input_item    = output_offset / block_threads;
      output[output_offset]   = input[input_tid * items_per_thread + input_item];
    }
  }
  return output;
}

template <typename T>
c2h::host_vector<T>
blocked_to_warp_striped_reference(const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  constexpr int warp_threads = cub::detail::warp_threads;

  c2h::host_vector<T> output(input.size());
  for (int tid = 0; tid < block_threads; ++tid)
  {
    const int warp_offset = (tid / warp_threads) * warp_threads * items_per_thread;
    const int lane_id     = tid % warp_threads;
    for (int item = 0; item < items_per_thread; ++item)
    {
      output[tid * items_per_thread + item] = input[warp_offset + item * warp_threads + lane_id];
    }
  }
  return output;
}

template <typename T>
c2h::host_vector<T>
warp_striped_to_blocked_reference(const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  constexpr int warp_threads = cub::detail::warp_threads;

  c2h::host_vector<T> output(input.size());
  for (int tid = 0; tid < block_threads; ++tid)
  {
    const int warp_offset   = (tid / warp_threads) * warp_threads * items_per_thread;
    const int lane_id       = tid % warp_threads;
    const int output_offset = lane_id * items_per_thread;
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int input_tid                   = (output_offset + item) % warp_threads;
      const int input_item                  = (output_offset + item) / warp_threads;
      output[tid * items_per_thread + item] = input[warp_offset + input_tid * items_per_thread + input_item];
    }
  }
  return output;
}

template <typename T>
c2h::host_vector<T> scatter_to_blocked_reference(const c2h::host_vector<T>& input)
{
  return c2h::host_vector<T>(input.rbegin(), input.rend());
}

template <typename T>
c2h::host_vector<T>
scatter_to_striped_reference(const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  c2h::host_vector<T> output(input.size());
  const int tile_size = block_threads * items_per_thread;
  for (int tid = 0; tid < block_threads; ++tid)
  {
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int output_offset = tid * items_per_thread + item;
      const int rank          = item * block_threads + tid;
      output[output_offset]   = input[tile_size - 1 - rank];
    }
  }
  return output;
}

template <typename T>
void require_valid_scatter_outputs(
  const c2h::device_vector<T>& output, const c2h::host_vector<T>& input, int block_threads, int items_per_thread)
{
  const c2h::host_vector<T> host_output = output;
  const int tile_size                   = block_threads * items_per_thread;

  for (int tid = 0; tid < block_threads; ++tid)
  {
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int output_offset = tid * items_per_thread + item;
      const int rank          = item * block_threads + tid;
      const int input_offset  = tile_size - 1 - rank;
      if (input_offset % 2 == 0)
      {
        REQUIRE(host_output[output_offset] == input[input_offset]);
      }
    }
  }
}

// %PARAM% ITEMS_PER_THREAD ipt 1:4:8

using types            = c2h::type_list<std::int32_t, std::int64_t>;
using block_threads    = c2h::enum_type_list<int, 32, 128>;
using items_per_thread = c2h::enum_type_list<int, ITEMS_PER_THREAD>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int block_threads    = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread = c2h::get<2, TestType>::value;
  static constexpr int tile_size        = block_threads * items_per_thread;
};

C2H_TEST("Block exchange striped to blocked works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    striped_to_blocked_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::striped_to_blocked, params::block_threads, params::items_per_thread>(data);

  REQUIRE(expected == data);
}

C2H_TEST("Block exchange blocked to striped works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    blocked_to_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::blocked_to_striped, params::block_threads, params::items_per_thread>(data);

  REQUIRE(expected == data);
}

C2H_TEST(
  "Block exchange warp-striped to blocked works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    warp_striped_to_blocked_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::warp_striped_to_blocked, params::block_threads, params::items_per_thread>(
    data);

  REQUIRE(expected == data);
}

C2H_TEST(
  "Block exchange blocked to warp-striped works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    blocked_to_warp_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::blocked_to_warp_striped, params::block_threads, params::items_per_thread>(
    data);

  REQUIRE(expected == data);
}

C2H_TEST("Block exchange scatter to blocked works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input    = data;
  const c2h::host_vector<type> expected = scatter_to_blocked_reference(input);

  invoke_block_exchange<block_exchange_op::scatter_to_blocked, params::block_threads, params::items_per_thread>(data);

  REQUIRE(expected == data);
}

C2H_TEST("Block exchange scatter to striped works in-place", "[exchange][block]", types, block_threads, items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    scatter_to_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::scatter_to_striped, params::block_threads, params::items_per_thread>(data);

  REQUIRE(expected == data);
}

C2H_TEST("Block exchange guarded scatter to striped works in-place",
         "[exchange][block]",
         types,
         block_threads,
         items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;

  invoke_block_exchange<block_exchange_op::scatter_to_striped_guarded, params::block_threads, params::items_per_thread>(
    data);

  require_valid_scatter_outputs(data, input, params::block_threads, params::items_per_thread);
}

C2H_TEST("Block exchange flagged scatter to striped works in-place",
         "[exchange][block]",
         types,
         block_threads,
         items_per_thread)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;

  invoke_block_exchange<block_exchange_op::scatter_to_striped_flagged, params::block_threads, params::items_per_thread>(
    data);

  require_valid_scatter_outputs(data, input, params::block_threads, params::items_per_thread);
}
