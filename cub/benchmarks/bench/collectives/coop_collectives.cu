// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce_broadcast.cuh>
#include <cub/block/block_row_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>
#include <cub/warp/warp_reduce_batched_broadcast.cuh>
#include <cub/warp/warp_reduce_broadcast.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_utils.cuh>

#include <cuda/std/array>

#include <cuda_runtime_api.h>
#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

using value_types = nvbench::type_list<float>;

template <typename T>
struct warp_reduce_owner_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    return warp_reduce_t{temp_storage[warp_id]}.Sum(thread_data);
  }
};

template <typename T>
struct warp_reduce_manual_broadcast_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id          = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    T aggregate                = warp_reduce_t{temp_storage[warp_id]}.Sum(thread_data);
    const auto logical_warp_id = cub::detail::logical_warp_id<cub::detail::warp_threads>();
    const auto member_mask     = cub::WarpMask<cub::detail::warp_threads>(logical_warp_id);
    return cub::ShuffleIndex<cub::detail::warp_threads>(aggregate, 0, member_mask);
  }
};

template <typename T>
struct warp_reduce_coop_broadcast_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBroadcast<T>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    return warp_reduce_t{temp_storage[warp_id]}.Sum(thread_data);
  }
};

template <typename T>
struct warp_allreduce4_manual_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    constexpr int logical_warp_threads = 4;
    const int lane_id                  = cub::detail::logical_lane_id<logical_warp_threads>();
    const int logical_warp_id          = cub::detail::logical_warp_id<logical_warp_threads>();
    const auto member_mask             = cub::WarpMask<logical_warp_threads>(logical_warp_id);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int offset = 1; offset < logical_warp_threads; offset <<= 1)
    {
      const T peer = cub::ShuffleIndex<logical_warp_threads>(thread_data, lane_id ^ offset, member_mask);
      thread_data += peer;
    }
    return thread_data;
  }
};

template <typename T>
struct warp_reduce_coop_broadcast4_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBroadcast<T, 4>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    return warp_reduce_t{temp_storage[warp_id]}.Sum(thread_data);
  }
};

template <typename T, int Batches>
struct warp_reduce_serial_batched_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduce<T>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[Batches][32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    const int lane_id = static_cast<int>(threadIdx.x) % cub::detail::warp_threads;

    T result = thread_data;
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      const T aggregate = warp_reduce_t{temp_storage[batch][warp_id]}.Sum(thread_data + static_cast<T>(batch));
      if (lane_id == batch)
      {
        result = aggregate;
      }
    }
    return result;
  }
};

template <typename T, int Batches>
struct warp_reduce_batched_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBatched<T, Batches>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    const int lane_id = static_cast<int>(threadIdx.x) % cub::detail::warp_threads;

    ::cuda::std::array<T, Batches> inputs{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      inputs[batch] = thread_data + static_cast<T>(batch);
    }

    T result = warp_reduce_t{temp_storage[warp_id]}.Sum(inputs);
    return lane_id < Batches ? result : thread_data;
  }
};

template <typename T, int Batches>
struct warp_reduce_serial_batched_broadcast4_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBroadcast<T, 4>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[Batches][32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;

    T result{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      result += warp_reduce_t{temp_storage[batch][warp_id]}.Sum(thread_data + static_cast<T>(batch));
    }
    return result;
  }
};

template <typename T, int Batches>
struct warp_reduce_batched_broadcast4_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_reduce_t = cub::WarpReduceBatchedBroadcast<T, Batches, 4>;
    __shared__ typename warp_reduce_t::TempStorage temp_storage[32];

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;

    ::cuda::std::array<T, Batches> inputs{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      inputs[batch] = thread_data + static_cast<T>(batch);
    }

    const auto outputs = warp_reduce_t{temp_storage[warp_id]}.Sum(inputs);
    return cub::ThreadReduce(outputs, ::cuda::std::plus<>{});
  }
};

template <typename T>
struct block_reduce_manual_broadcast_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using block_reduce_t = cub::BlockReduce<T, 256>;
    struct temp_storage_t
    {
      typename block_reduce_t::TempStorage reduce;
      cub::Uninitialized<T> aggregate;
    };

    __shared__ temp_storage_t temp_storage;

    T aggregate = block_reduce_t{temp_storage.reduce}.Sum(thread_data);
    if (threadIdx.x == 0)
    {
      temp_storage.aggregate.Alias() = aggregate;
    }
    __syncthreads();

    T result = temp_storage.aggregate.Alias();
    __syncthreads();
    return result;
  }
};

template <typename T>
struct block_reduce_coop_broadcast_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using block_reduce_t = cub::BlockReduceBroadcast<T, 256>;
    __shared__ typename block_reduce_t::TempStorage temp_storage;

    return block_reduce_t{temp_storage}.Sum(thread_data);
  }
};

template <typename T>
struct block_row_reduce_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using row_reduce_t = cub::BlockRowReduce<T, 4, 2>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;

    return row_reduce_t{temp_storage}.Sum(thread_data);
  }
};

template <typename T>
struct block_row_reduce_warp_broadcast_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using row_reduce_t = cub::BlockRowReduceWarpBroadcast<T, 4, 2>;
    __shared__ typename row_reduce_t::TempStorage temp_storage;

    return row_reduce_t{temp_storage}.Sum(thread_data);
  }
};

template <typename T>
struct warp_block_scan_t
{
  _CCCL_DEVICE _CCCL_FORCEINLINE T operator()(T thread_data) const
  {
    using warp_scan_t  = cub::WarpScan<T>;
    using block_scan_t = cub::BlockScan<T, 256>;

    __shared__ typename warp_scan_t::TempStorage warp_storage[8];
    __shared__ typename block_scan_t::TempStorage block_storage;

    const int warp_id = static_cast<int>(threadIdx.x) / cub::detail::warp_threads;
    T broadcast       = warp_scan_t{warp_storage[warp_id]}.Broadcast(thread_data, 0);

    T prefix{};
    block_scan_t{block_storage}.ExclusiveSum(static_cast<T>(1), prefix);
    return broadcast + prefix;
  }
};

template <int BlockSize, int UnrollFactor, template <typename> class ActionT, typename T>
void bench_collective(nvbench::state& state, nvbench::type_list<T>)
{
  using action_t          = ActionT<T>;
  const auto& kernel      = benchmark_kernel<BlockSize, UnrollFactor, action_t, T>;
  const int num_sms       = state.get_device().value().get_number_of_sms();
  int max_blocks_per_sm   = 0;
  const std::size_t smem  = 0;
  const int block_threads = BlockSize;
  const int unroll_factor = UnrollFactor;

  NVBENCH_CUDA_CALL_NOEXCEPT(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, kernel, block_threads, smem));

  const int grid_size = max_blocks_per_sm * num_sms;
  state.add_element_count(grid_size * block_threads * unroll_factor, "Thread ops");
  state.add_element_count(grid_size * unroll_factor, "CTA ops");
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch&) {
    kernel<<<grid_size, block_threads>>>(action_t{});
  });
}

template <typename T>
using warp_reduce_serial_batched_4_t = warp_reduce_serial_batched_t<T, 4>;

template <typename T>
using warp_reduce_batched_4_t = warp_reduce_batched_t<T, 4>;

template <typename T>
using warp_reduce_serial_batched_broadcast4_4_t = warp_reduce_serial_batched_broadcast4_t<T, 4>;

template <typename T>
using warp_reduce_batched_broadcast4_4_t = warp_reduce_batched_broadcast4_t<T, 4>;

template <typename T>
void warp_reduce_owner(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_owner_t>(state, type);
}

template <typename T>
void warp_reduce_manual_broadcast(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_manual_broadcast_t>(state, type);
}

template <typename T>
void warp_reduce_coop_broadcast(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_coop_broadcast_t>(state, type);
}

template <typename T>
void warp_allreduce4_manual(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_allreduce4_manual_t>(state, type);
}

template <typename T>
void warp_reduce_coop_broadcast4(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_coop_broadcast4_t>(state, type);
}

template <typename T>
void warp_reduce_serial_batched_4(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_serial_batched_4_t>(state, type);
}

template <typename T>
void warp_reduce_batched_4(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_batched_4_t>(state, type);
}

template <typename T>
void warp_reduce_serial_batched_broadcast4_4(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_serial_batched_broadcast4_4_t>(state, type);
}

template <typename T>
void warp_reduce_batched_broadcast4_4(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 128, warp_reduce_batched_broadcast4_4_t>(state, type);
}

template <typename T>
void block_reduce_manual_broadcast(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 64, block_reduce_manual_broadcast_t>(state, type);
}

template <typename T>
void block_reduce_coop_broadcast(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 64, block_reduce_coop_broadcast_t>(state, type);
}

template <typename T>
void block_row_reduce(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 64, block_row_reduce_t>(state, type);
}

template <typename T>
void block_row_reduce_warp_broadcast(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 64, block_row_reduce_warp_broadcast_t>(state, type);
}

template <typename T>
void warp_block_scan(nvbench::state& state, nvbench::type_list<T> type)
{
  bench_collective<256, 64, warp_block_scan_t>(state, type);
}

NVBENCH_BENCH_TYPES(warp_reduce_owner, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_owner")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_manual_broadcast, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_manual_broadcast")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_coop_broadcast, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_coop_broadcast")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_allreduce4_manual, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_allreduce4_manual")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_coop_broadcast4, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_coop_broadcast4")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_serial_batched_4, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_serial_batched_4")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_batched_4, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_batched_4")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_serial_batched_broadcast4_4, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_serial_batched_broadcast4_4")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_reduce_batched_broadcast4_4, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_reduce_batched_broadcast4_4")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(block_reduce_manual_broadcast, NVBENCH_TYPE_AXES(value_types))
  .set_name("block_reduce_manual_broadcast")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(block_reduce_coop_broadcast, NVBENCH_TYPE_AXES(value_types))
  .set_name("block_reduce_coop_broadcast")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(block_row_reduce, NVBENCH_TYPE_AXES(value_types))
  .set_name("block_row_reduce")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(block_row_reduce_warp_broadcast, NVBENCH_TYPE_AXES(value_types))
  .set_name("block_row_reduce_warp_broadcast")
  .set_type_axes_names({"T{ct}"});

NVBENCH_BENCH_TYPES(warp_block_scan, NVBENCH_TYPE_AXES(value_types))
  .set_name("warp_block_scan")
  .set_type_axes_names({"T{ct}"});
