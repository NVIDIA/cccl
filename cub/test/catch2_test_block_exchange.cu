// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/block/block_exchange.cuh>

#include <thrust/memory.h>

#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/type_traits>

#include <cstdint>

#include "cub_test_macros.h"
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

template <block_exchange_op Op, int BlockThreads, int ItemsPerThread, bool WarpTimeSlicing, typename T>
__global__ void block_exchange_kernel(T* data)
{
  using block_exchange_t = cub::BlockExchange<T, BlockThreads, ItemsPerThread, WarpTimeSlicing>;
  using temp_storage_t   = typename block_exchange_t::TempStorage;

  __shared__ temp_storage_t temp_storage;

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
    constexpr int tile_size = BlockThreads * ItemsPerThread;

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

template <block_exchange_op Op, int BlockThreads, int ItemsPerThread, bool WarpTimeSlicing, typename T>
void invoke_block_exchange(c2h::device_vector<T>& data)
{
  block_exchange_kernel<Op, BlockThreads, ItemsPerThread, WarpTimeSlicing>
    <<<1, BlockThreads>>>(thrust::raw_pointer_cast(data.data()));

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
      output[(tid * items_per_thread) + item] = input[(item * block_threads) + tid];
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
      const int output_offset = (tid * items_per_thread) + item;
      const int input_tid     = output_offset % block_threads;
      const int input_item    = output_offset / block_threads;
      output[output_offset]   = input[(input_tid * items_per_thread) + input_item];
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
      output[(tid * items_per_thread) + item] = input[warp_offset + (item * warp_threads) + lane_id];
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
      const int input_tid                     = (output_offset + item) % warp_threads;
      const int input_item                    = (output_offset + item) / warp_threads;
      output[(tid * items_per_thread) + item] = input[warp_offset + (input_tid * items_per_thread) + input_item];
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
      const int output_offset = (tid * items_per_thread) + item;
      const int rank          = (item * block_threads) + tid;
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
      const int output_offset = (tid * items_per_thread) + item;
      const int rank          = (item * block_threads) + tid;
      const int input_offset  = tile_size - 1 - rank;
      if (input_offset % 2 == 0)
      {
        REQUIRE(host_output[output_offset] == input[input_offset]);
      }
    }
  }
}

// %PARAM% TEST_TYPES types 0:1

#if TEST_TYPES == 0
using types = c2h::type_list<std::int32_t>;
#elif TEST_TYPES == 1
using types = c2h::type_list<std::int64_t>;
#endif

using block_threads     = c2h::enum_type_list<int, 32, 128>;
using items_per_thread  = c2h::enum_type_list<int, 1, 4, 5, 8>;
using warp_time_slicing = c2h::enum_type_list<bool, false, true>;

// Warp time-slicing is unsupported for guarded and flagged scatter exchanges:
// https://github.com/NVIDIA/cccl/issues/870
using no_warp_time_slicing = c2h::enum_type_list<bool, false>;

template <typename TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int block_threads      = c2h::get<1, TestType>::value;
  static constexpr int items_per_thread   = c2h::get<2, TestType>::value;
  static constexpr bool warp_time_slicing = c2h::get<3, TestType>::value;
  static constexpr int tile_size          = block_threads * items_per_thread;
};

CUB_TEST("Block exchange striped to blocked works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    striped_to_blocked_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::striped_to_blocked,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange blocked to striped works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    blocked_to_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::blocked_to_striped,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange warp-striped to blocked works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    warp_striped_to_blocked_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::warp_striped_to_blocked,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange blocked to warp-striped works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    blocked_to_warp_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::blocked_to_warp_striped,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange scatter to blocked works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input    = data;
  const c2h::host_vector<type> expected = scatter_to_blocked_reference(input);

  invoke_block_exchange<block_exchange_op::scatter_to_blocked,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange scatter to striped works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;
  const c2h::host_vector<type> expected =
    scatter_to_striped_reference(input, params::block_threads, params::items_per_thread);

  invoke_block_exchange<block_exchange_op::scatter_to_striped,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  REQUIRE(expected == data);
}

CUB_TEST("Block exchange guarded scatter to striped works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         no_warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;

  invoke_block_exchange<block_exchange_op::scatter_to_striped_guarded,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  require_valid_scatter_outputs(data, input, params::block_threads, params::items_per_thread);
}

CUB_TEST("Block exchange flagged scatter to striped works in-place",
         "[exchange][block]",
         CUB_SMALL,
         types,
         block_threads,
         items_per_thread,
         no_warp_time_slicing)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  c2h::device_vector<type> data(params::tile_size);
  c2h::gen(C2H_SEED(1), data);

  const c2h::host_vector<type> input = data;

  invoke_block_exchange<block_exchange_op::scatter_to_striped_flagged,
                        params::block_threads,
                        params::items_per_thread,
                        params::warp_time_slicing>(data);

  require_valid_scatter_outputs(data, input, params::block_threads, params::items_per_thread);
}

#if TEST_TYPES == 0
template <int BlockThreads, bool WarpTimeSlicing, typename InputT, typename OutputT>
__global__ void single_item_exchange_transpose_kernel(
  const InputT* d_in,
  InputT* d_in_place,
  InputT* d_aliased,
  OutputT* d_striped_to_blocked,
  OutputT* d_blocked_to_striped,
  OutputT* d_warp_striped_to_blocked,
  OutputT* d_blocked_to_warp_striped)
{
  using block_exchange_t = cub::BlockExchange<InputT, BlockThreads, 1, WarpTimeSlicing>;
  __shared__ typename block_exchange_t::TempStorage temp_storage;
  block_exchange_t exchange(temp_storage);

  const int tid = static_cast<int>(threadIdx.x);

  InputT in_place[1] = {d_in[tid]};
  exchange.StripedToBlocked(in_place);
  exchange.BlockedToStriped(in_place);
  exchange.WarpStripedToBlocked(in_place);
  exchange.BlockedToWarpStriped(in_place);
  d_in_place[tid] = in_place[0];

  InputT aliased[1] = {d_in[tid]};
  exchange.StripedToBlocked(aliased, aliased);
  exchange.BlockedToStriped(aliased, aliased);
  exchange.WarpStripedToBlocked(aliased, aliased);
  exchange.BlockedToWarpStriped(aliased, aliased);
  d_aliased[tid] = aliased[0];

  const InputT src[1] = {d_in[tid]};
  OutputT out0[1]{};
  OutputT out1[1]{};
  OutputT out2[1]{};
  OutputT out3[1]{};
  exchange.StripedToBlocked(src, out0);
  exchange.BlockedToStriped(src, out1);
  exchange.WarpStripedToBlocked(src, out2);
  exchange.BlockedToWarpStriped(src, out3);

  d_striped_to_blocked[tid]      = out0[0];
  d_blocked_to_striped[tid]      = out1[0];
  d_warp_striped_to_blocked[tid] = out2[0];
  d_blocked_to_warp_striped[tid] = out3[0];
}

template <bool WarpTimeSlicing>
void test_single_item_exchange_transpose()
{
  constexpr int block_threads = 128;

  c2h::device_vector<int> d_in(block_threads);
  c2h::gen(C2H_SEED(1), d_in);

  c2h::device_vector<int> d_in_place(block_threads, 0);
  c2h::device_vector<int> d_aliased(block_threads, 0);
  c2h::device_vector<long long> d_s2b(block_threads, 0);
  c2h::device_vector<long long> d_b2s(block_threads, 0);
  c2h::device_vector<long long> d_ws2b(block_threads, 0);
  c2h::device_vector<long long> d_b2ws(block_threads, 0);

  single_item_exchange_transpose_kernel<block_threads, WarpTimeSlicing, int, long long><<<1, block_threads>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_in_place.data()),
    thrust::raw_pointer_cast(d_aliased.data()),
    thrust::raw_pointer_cast(d_s2b.data()),
    thrust::raw_pointer_cast(d_b2s.data()),
    thrust::raw_pointer_cast(d_ws2b.data()),
    thrust::raw_pointer_cast(d_b2ws.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_in_place == d_in);
  REQUIRE(d_aliased == d_in);

  c2h::host_vector<long long> expected(d_in.size());
  c2h::host_vector<int> h_in = d_in;
  for (std::size_t i = 0; i < h_in.size(); ++i)
  {
    expected[i] = static_cast<long long>(h_in[i]);
  }
  REQUIRE(c2h::host_vector<long long>(d_s2b) == expected);
  REQUIRE(c2h::host_vector<long long>(d_b2s) == expected);
  REQUIRE(c2h::host_vector<long long>(d_ws2b) == expected);
  REQUIRE(c2h::host_vector<long long>(d_b2ws) == expected);
}

struct assign_only_exchange_output
{
  int value{};

  __host__ __device__ assign_only_exchange_output() = default;
  __host__ __device__ assign_only_exchange_output(const assign_only_exchange_output& other)
      : value(other.value)
  {}
  assign_only_exchange_output(int) = delete;

  __host__ __device__ assign_only_exchange_output& operator=(int v)
  {
    value = v;
    return *this;
  }
  __host__ __device__ assign_only_exchange_output& operator=(const assign_only_exchange_output& other)
  {
    value = other.value;
    return *this;
  }
};

struct construct_only_exchange_output
{
  int value{};

  __host__ __device__ construct_only_exchange_output() = default;
  __host__ __device__ construct_only_exchange_output(int v)
      : value(v)
  {}
  __host__ __device__ construct_only_exchange_output(const construct_only_exchange_output& other)
      : value(other.value)
  {}
  construct_only_exchange_output& operator=(int)                                   = delete;
  construct_only_exchange_output& operator=(const construct_only_exchange_output&) = delete;
};

template <int BlockThreads, bool WarpTimeSlicing>
__global__ void
single_item_exchange_conversion_semantics_kernel(const int* d_in, int* d_s2b, int* d_b2s, int* d_b2ws, int* d_ws2b)
{
  using block_exchange_t = cub::BlockExchange<int, BlockThreads, 1, WarpTimeSlicing>;
  __shared__ typename block_exchange_t::TempStorage temp_storage;
  block_exchange_t exchange(temp_storage);

  const int tid    = static_cast<int>(threadIdx.x);
  const int src[1] = {d_in[tid]};

  assign_only_exchange_output s2b_out[1]{};
  assign_only_exchange_output b2s_out[1]{};
  assign_only_exchange_output b2ws_out[1]{};
  exchange.StripedToBlocked(src, s2b_out);
  exchange.BlockedToStriped(src, b2s_out);
  exchange.BlockedToWarpStriped(src, b2ws_out);
  d_s2b[tid]  = s2b_out[0].value;
  d_b2s[tid]  = b2s_out[0].value;
  d_b2ws[tid] = b2ws_out[0].value;

  if constexpr (WarpTimeSlicing)
  {
    assign_only_exchange_output ws2b_out[1]{};
    exchange.WarpStripedToBlocked(src, ws2b_out);
    d_ws2b[tid] = ws2b_out[0].value;
  }
  else
  {
    construct_only_exchange_output ws2b_out[1]{};
    exchange.WarpStripedToBlocked(src, ws2b_out);
    d_ws2b[tid] = ws2b_out[0].value;
  }
}

template <bool WarpTimeSlicing>
void test_single_item_exchange_conversion_semantics()
{
  constexpr int block_threads = 128;

  c2h::device_vector<int> d_in(block_threads);
  c2h::gen(C2H_SEED(1), d_in);

  c2h::device_vector<int> d_s2b(block_threads, 0);
  c2h::device_vector<int> d_b2s(block_threads, 0);
  c2h::device_vector<int> d_b2ws(block_threads, 0);
  c2h::device_vector<int> d_ws2b(block_threads, 0);

  single_item_exchange_conversion_semantics_kernel<block_threads, WarpTimeSlicing><<<1, block_threads>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_s2b.data()),
    thrust::raw_pointer_cast(d_b2s.data()),
    thrust::raw_pointer_cast(d_b2ws.data()),
    thrust::raw_pointer_cast(d_ws2b.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_s2b == d_in);
  REQUIRE(d_b2s == d_in);
  REQUIRE(d_b2ws == d_in);
  REQUIRE(d_ws2b == d_in);
}

struct self_resetting_exchange_item
{
  int value{};

  __host__ __device__ self_resetting_exchange_item() = default;
  __host__ __device__ explicit self_resetting_exchange_item(int v)
      : value(v)
  {}
  __host__ __device__ self_resetting_exchange_item(const self_resetting_exchange_item& other)
      : value(0)
  {
    value = other.value;
  }
  self_resetting_exchange_item& operator=(const self_resetting_exchange_item&) = delete;
};

template <int BlockThreads>
__global__ void
single_item_warp_striped_aliased_non_trivial_kernel(const int* d_in, int* d_aliased_out, int* d_distinct_out)
{
  static_assert(!::cuda::is_trivially_copyable_v<self_resetting_exchange_item>);
  using block_exchange_t = cub::BlockExchange<self_resetting_exchange_item, BlockThreads, 1, false>;
  __shared__ typename block_exchange_t::TempStorage temp_storage;
  block_exchange_t exchange(temp_storage);

  const int tid = static_cast<int>(threadIdx.x);
  self_resetting_exchange_item aliased[1]{self_resetting_exchange_item{d_in[tid]}};
  exchange.WarpStripedToBlocked(aliased);
  exchange.WarpStripedToBlocked(aliased, aliased);
  d_aliased_out[tid] = aliased[0].value;

  self_resetting_exchange_item distinct[1]{};
  exchange.WarpStripedToBlocked(aliased, distinct);
  d_distinct_out[tid] = distinct[0].value;
}

void test_single_item_warp_striped_aliased_non_trivial()
{
  constexpr int block_threads = 128;

  c2h::device_vector<int> d_in(block_threads);
  c2h::gen(C2H_SEED(1), d_in);

  c2h::device_vector<int> d_aliased_out(block_threads, 0);
  c2h::device_vector<int> d_distinct_out(block_threads, 0);

  single_item_warp_striped_aliased_non_trivial_kernel<block_threads><<<1, block_threads>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_aliased_out.data()),
    thrust::raw_pointer_cast(d_distinct_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  REQUIRE(d_aliased_out == d_in);
  REQUIRE(d_distinct_out == d_in);
}

CUB_TEST_CASE("BlockExchange single-item transposes handle in-place, aliased, and cross-type outputs",
              "[exchange][block]",
              CUB_SMALL)
{
  // Scatter exchanges still require TempStorage when ItemsPerThread is 1.
  STATIC_REQUIRE(
    !::cuda::std::is_same_v<typename cub::BlockExchange<int, 128, 1>::TempStorage, cub::Uninitialized<cub::NullType>>);
  STATIC_REQUIRE(
    sizeof(typename cub::BlockExchange<int, 128, 1>::TempStorage) > sizeof(cub::Uninitialized<cub::NullType>));

  test_single_item_exchange_transpose<false>();
  test_single_item_exchange_transpose<true>();
  test_single_item_exchange_conversion_semantics<false>();
  test_single_item_exchange_conversion_semantics<true>();
  test_single_item_warp_striped_aliased_non_trivial();
}
#endif // TEST_TYPES == 0
