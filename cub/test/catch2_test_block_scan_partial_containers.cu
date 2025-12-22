// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_scan.cuh>

#include <cuda/functional>
#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <climits>

#include "catch2_test_block_scan_partial_helper.cuh"
#include <c2h/catch2_test_helper.h>

enum class container
{
  array,
  span,
  mdspan
};

template <int ItemsPerThread, int BlockDim, class T, class ActionT>
__global__ void block_scan_kernel_array(T* in, T* out, ActionT action, int valid_items)
{
  using block_scan_t = cub::BlockScan<T, BlockDim>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  cuda::std::array<T, ItemsPerThread> thread_data;

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDim, 1, 1));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }

  block_scan_t scan(storage);

  action(scan, thread_data, valid_items);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = thread_data[item];
  }
}

template <int ItemsPerThread, int BlockDim, class T, class ActionT>
__global__ void block_scan_kernel_span(T* in, T* out, ActionT action, int valid_items)
{
  using block_scan_t = cub::BlockScan<T, BlockDim>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  T thread_data[ItemsPerThread];

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDim, 1, 1));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }

  block_scan_t scan(storage);

  cuda::std::span<T, ItemsPerThread> span{thread_data};
  action(scan, span, valid_items);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = thread_data[item];
  }
}

#if _CCCL_STD_VER >= 2023

template <int ItemsPerThread, int BlockDim, class T, class ActionT>
__global__ void block_scan_kernel_mdspan(T* in, T* out, ActionT action, int valid_items)
{
  using block_scan_t = cub::BlockScan<T, BlockDim>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  T thread_data[ItemsPerThread];

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDim, 1, 1));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }

  block_scan_t scan(storage);

  using Extents = cuda::std::extents<int, ItemsPerThread>;
  cuda::std::mdspan<T, Extents> mdspan{thread_data, Extents{}};
  action(scan, mdspan, valid_items);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = thread_data[item];
  }
}

#endif // _CCCL_STD_VER >= 2023

template <container cont, int ItemsPerThread, int BlockDim, class T, class ActionT>
void block_scan_container(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action, int valid_items)
{
  dim3 block_dims(BlockDim);

  switch (cont)
  {
    case container::array:
      block_scan_kernel_array<ItemsPerThread, BlockDim, T, ActionT><<<1, block_dims>>>(
        thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);
      break;
    case container::span:
      block_scan_kernel_span<ItemsPerThread, BlockDim, T, ActionT><<<1, block_dims>>>(
        thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);
      break;
#if _CCCL_STD_VER >= 2023
    case container::mdspan:
      block_scan_kernel_mdspan<ItemsPerThread, BlockDim, T, ActionT><<<1, block_dims>>>(
        thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);
      break;
#endif // _CCCL_STD_VER >= 2023
  }

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename InT, typename OutT>
struct custom_mixed_min
{
  __host__ __device__ OutT operator()(InT left, InT right) const
  {
    return cuda::std::min<OutT>(left, right);
  }

  // Make sure that we are always accumulating to InT
  __host__ __device__ OutT operator()(OutT, InT) const = delete;
};

template <typename T, scan_mode Mode, typename OpValT>
struct min_mixed_op_t
{
  template <typename Container, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, Container& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode, typename OpValT>
struct min_mixed_init_value_op_t
{
  T initial_value;
  template <typename Container, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, Container& thread_data, int valid_items) const
  {
    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, initial_value, custom_mixed_min<T, OpValT>{}, valid_items);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, initial_value, custom_mixed_min<T, OpValT>{}, valid_items);
    }
  }
};

template <class T, scan_mode Mode, typename OpValT>
struct min_mixed_init_value_aggregate_op_t
{
  int m_target_thread_id;
  T initial_value;
  T* m_d_block_aggregate;

  template <typename Container, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, Container& thread_data, int valid_items) const
  {
    T block_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(
        thread_data, thread_data, initial_value, ::custom_mixed_min<T, OpValT>{}, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(
        thread_data, thread_data, initial_value, ::custom_mixed_min<T, OpValT>{}, valid_items, block_aggregate);
    }

    const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode, typename OpValT>
struct min_mixed_aggregate_op_t
{
  int m_target_thread_id;
  T* m_d_block_aggregate;

  template <typename Container, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, Container& thread_data, int valid_items) const
  {
    T block_aggregate{};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(
        thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items, block_aggregate);
    }
    else
    {
      scan.InclusiveScanPartialTile(
        thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items, block_aggregate);
    }

    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));

    if (tid == m_target_thread_id)
    {
      *m_d_block_aggregate = block_aggregate;
    }
  }
};

template <class T, scan_mode Mode, typename OpValT>
struct min_mixed_prefix_op_t
{
  T m_prefix;
  static constexpr T min_identity = cuda::std::numeric_limits<T>::max();

  struct block_prefix_op_t
  {
    int linear_tid;
    T prefix;

    __device__ block_prefix_op_t(int linear_tid, T prefix)
        : linear_tid(linear_tid)
        , prefix(prefix)
    {}

    __device__ T operator()(T block_aggregate)
    {
      T retval = (linear_tid == 0) ? prefix : min_identity;
      prefix   = custom_mixed_min<T, OpValT>{}(prefix, block_aggregate);
      return retval;
    }
  };

  template <typename Container, class BlockScanT>
  __device__ void operator()(BlockScanT& scan, Container& thread_data, int valid_items) const
  {
    const int tid = static_cast<int>(cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z));
    block_prefix_op_t prefix_op{tid, m_prefix};

    if constexpr (Mode == scan_mode::exclusive)
    {
      scan.ExclusiveScanPartialTile(thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items, prefix_op);
    }
    else
    {
      scan.InclusiveScanPartialTile(thread_data, thread_data, custom_mixed_min<T, OpValT>{}, valid_items, prefix_op);
    }
  }
};

// %PARAM% TEST_MODE mode 0:1

#if TEST_MODE == 0
using modes = c2h::enum_type_list<scan_mode, scan_mode::inclusive>;
#else
using modes = c2h::enum_type_list<scan_mode, scan_mode::exclusive>;
#endif

using containers =
  c2h::enum_type_list<container,
                      container::array,
                      container::span
#if _CCCL_STD_VER >= 2023
                      ,
                      container::mdspan
#endif // _CCCL_STD_VER >= 2023
                      >;

template <class TestType>
struct mode_params_t
{
  using type          = cuda::std::int32_t;
  using op_value_type = cuda::std::int64_t;

  static constexpr int threads_in_block = 256;
  static constexpr int items_per_thread = 3;
  static constexpr int tile_size        = items_per_thread * threads_in_block;

  static constexpr scan_mode mode = c2h::get<0, TestType>::value;
  static constexpr container cont = c2h::get<1, TestType>::value;
};

C2H_TEST("Partial block scan supports containers", "[scan][block]", modes, containers)
{
  using params          = mode_params_t<TestType>;
  using type            = typename params::type;
  using op_value_type   = typename params::op_value_type;
  using valid_items_gen = valid_items_generators_t<params>;

  const int valid_items = GENERATE_COPY(valid_items_gen::rand_inside());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(1), d_in);

  block_scan_container<params::cont, params::items_per_thread, params::threads_in_block>(
    d_in, d_out, min_mixed_op_t<type, params::mode, op_value_type>{}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    params::mode,
    h_out,
    [](type l, type r) {
      return std::min(static_cast<op_value_type>(l), static_cast<op_value_type>(r));
    },
    valid_items,
    INT_MAX);

  if constexpr (params::mode == scan_mode::exclusive)
  {
    //! With no initial value, the output computed for *thread*\ :sub:`0` is undefined.
    d_out.erase(d_out.begin());
    h_out.erase(h_out.begin());
  }

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan supports containers and returns valid block aggregate", "[scan][block]", modes, containers)
{
  using params          = mode_params_t<TestType>;
  using type            = typename params::type;
  using op_value_type   = typename params::op_value_type;
  using valid_items_gen = valid_items_generators_t<params>;

  const int target_thread_id = GENERATE_COPY(take(1, random(0, params::threads_in_block - 1)));

  const int valid_items = GENERATE_COPY(valid_items_gen::rand_inside());
  c2h::device_vector<type> d_block_aggregate(1);
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(1), d_in);

  block_scan_container<params::cont, params::items_per_thread, params::threads_in_block>(
    d_in,
    d_out,
    min_mixed_aggregate_op_t<type, params::mode, op_value_type>{
      target_thread_id, thrust::raw_pointer_cast(d_block_aggregate.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;
  type block_aggregate         = host_scan(
    params::mode,
    h_out,
    [](type l, type r) {
      return std::min(static_cast<op_value_type>(l), static_cast<op_value_type>(r));
    },
    valid_items,
    INT_MAX);

  if constexpr (params::mode == scan_mode::exclusive)
  {
    // Undefined
    h_out[0] = d_out[0];
  }
  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan supports containers and works with initial value", "[scan][block]", modes, containers)
{
  using params          = mode_params_t<TestType>;
  using type            = typename params::type;
  using op_value_type   = typename params::op_value_type;
  using valid_items_gen = valid_items_generators_t<params>;

  const int valid_items = GENERATE_COPY(valid_items_gen::rand_inside());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(1), d_in);

  const auto initial_value = static_cast<type>(GENERATE_COPY(take(1, random(0, params::tile_size))));

  block_scan_container<params::cont, params::items_per_thread, params::threads_in_block>(
    d_in, d_out, min_mixed_init_value_op_t<type, params::mode, op_value_type>{initial_value}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    params::mode,
    h_out,
    [](type l, type r) {
      return std::min(static_cast<op_value_type>(l), static_cast<op_value_type>(r));
    },
    valid_items,
    initial_value);

  REQUIRE(h_out == d_out);
}

C2H_TEST("Partial block scan with initial value supports containers and returns valid block aggregate",
         "[scan][block]",
         modes,
         containers)
{
  using params          = mode_params_t<TestType>;
  using type            = typename params::type;
  using op_value_type   = typename params::op_value_type;
  using valid_items_gen = valid_items_generators_t<params>;

  const int valid_items = GENERATE_COPY(valid_items_gen::rand_above());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(1), d_in, 0, 1);

  const auto initial_value = static_cast<type>(GENERATE_COPY(take(2, random(0, params::tile_size))));

  const int target_thread_id = GENERATE_COPY(take(1, random(0, params::threads_in_block - 1)));
  CAPTURE(valid_items, initial_value, target_thread_id, params::tile_size, d_in);

  c2h::device_vector<type> d_block_aggregate(1);

  block_scan_container<params::cont, params::items_per_thread, params::threads_in_block>(
    d_in,
    d_out,
    min_mixed_init_value_aggregate_op_t<type, params::mode, op_value_type>{
      target_thread_id, initial_value, thrust::raw_pointer_cast(d_block_aggregate.data())},
    valid_items);

  c2h::host_vector<type> h_out = d_in;
  type h_block_aggregate       = host_scan(
    params::mode,
    h_out,
    [](type l, type r) {
      return std::min(static_cast<op_value_type>(l), static_cast<op_value_type>(r));
    },
    valid_items,
    initial_value);

  REQUIRE(h_out == d_out);
  if (valid_items > 0)
  {
    REQUIRE(h_block_aggregate == d_block_aggregate[0]);
  }
}

C2H_TEST("Partial block scan supports prefix op and containers", "[scan][block]", modes, containers)
{
  using params          = mode_params_t<TestType>;
  using type            = typename params::type;
  using op_value_type   = typename params::op_value_type;
  using valid_items_gen = valid_items_generators_t<params>;

  const type prefix = GENERATE_COPY(take(1, random(0, params::tile_size)));

  const int valid_items = GENERATE_COPY(valid_items_gen::rand_inside());
  c2h::device_vector<type> d_out(params::tile_size);
  c2h::device_vector<type> d_in(params::tile_size);
  c2h::gen(C2H_SEED(1), d_in);

  block_scan_container<params::cont, params::items_per_thread, params::threads_in_block>(
    d_in, d_out, min_mixed_prefix_op_t<type, params::mode, op_value_type>{prefix}, valid_items);

  c2h::host_vector<type> h_out = d_in;
  host_scan(
    params::mode,
    h_out,
    [](type l, type r) {
      return std::min(static_cast<op_value_type>(l), static_cast<op_value_type>(r));
    },
    valid_items,
    prefix);

  REQUIRE(h_out == d_out);
}
