// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_broadcast.cuh>

#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/operator.cuh>

inline constexpr int warp_size                  = 32;
inline constexpr int block_size                 = 2 * warp_size;
inline constexpr int num_items_per_thread       = 4;
inline constexpr int mixed_logical_warp_threads = 8;

enum class WarpReduceBroadcastMode
{
  Sum,
  SumValidItems,
  SumMultipleItems,
  Max,
  MaxValidItems,
  MaxMultipleItems,
  Min,
  MinValidItems,
  MinMultipleItems,
  Reduce,
  ReduceValidItems,
  ReduceMultipleItems
};

template <WarpReduceBroadcastMode Mode>
inline constexpr int items_per_thread_for_mode_v =
  (Mode == WarpReduceBroadcastMode::SumMultipleItems || Mode == WarpReduceBroadcastMode::MaxMultipleItems
   || Mode == WarpReduceBroadcastMode::MinMultipleItems || Mode == WarpReduceBroadcastMode::ReduceMultipleItems)
    ? num_items_per_thread
    : 1;

template <WarpReduceBroadcastMode Mode>
inline constexpr bool is_partial_mode_v =
  (Mode == WarpReduceBroadcastMode::SumValidItems || Mode == WarpReduceBroadcastMode::MaxValidItems
   || Mode == WarpReduceBroadcastMode::MinValidItems || Mode == WarpReduceBroadcastMode::ReduceValidItems);

struct affine_value_t
{
  int scale;
  int offset;

  friend bool operator==(const affine_value_t& lhs, const affine_value_t& rhs)
  {
    return lhs.scale == rhs.scale && lhs.offset == rhs.offset;
  }
};

struct affine_compose_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE affine_value_t operator()(affine_value_t lhs, affine_value_t rhs) const
  {
    constexpr int modulo = 32749;
    return affine_value_t{(lhs.scale * rhs.scale) % modulo, (lhs.scale * rhs.offset + lhs.offset) % modulo};
  }
};

template <>
inline constexpr affine_value_t identity_v<affine_compose_op, affine_value_t> = affine_value_t{1, 0};

inline constexpr int packed_affine_modulo = 251;

_CCCL_HOST_DEVICE constexpr int pack_affine_value(int scale, int offset)
{
  return scale * packed_affine_modulo + offset;
}

struct packed_affine_compose_op
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(int lhs, int rhs) const
  {
    const int lhs_scale  = lhs / packed_affine_modulo;
    const int lhs_offset = lhs % packed_affine_modulo;
    const int rhs_scale  = rhs / packed_affine_modulo;
    const int rhs_offset = rhs % packed_affine_modulo;
    const int scale      = (lhs_scale * rhs_scale) % packed_affine_modulo;
    const int offset     = (lhs_scale * rhs_offset + lhs_offset) % packed_affine_modulo;
    return pack_affine_value(scale, offset);
  }
};

template <>
inline constexpr int identity_v<packed_affine_compose_op, int> = pack_affine_value(1, 0);

template <WarpReduceBroadcastMode Mode, int LogicalWarpThreads, int ItemsPerThread, typename T, typename ReductionOp>
__global__ void warp_reduce_broadcast_kernel(const T* input, T* output, ReductionOp reduction_op, int valid_items)
{
  using warp_reduce_t = cub::WarpReduceBroadcast<T, LogicalWarpThreads>;

  __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

  const int tid                   = static_cast<int>(threadIdx.x);
  const int block_logical_warp_id = tid / LogicalWarpThreads;

  if constexpr (Mode == WarpReduceBroadcastMode::Sum)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Sum(input[tid]);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::SumValidItems)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Sum(input[tid], valid_items);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::Max)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Max(input[tid]);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::MaxValidItems)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Max(input[tid], valid_items);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::Min)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Min(input[tid]);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::MinValidItems)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Min(input[tid], valid_items);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::Reduce)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Reduce(input[tid], reduction_op);
  }
  else if constexpr (Mode == WarpReduceBroadcastMode::ReduceValidItems)
  {
    output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Reduce(input[tid], reduction_op, valid_items);
  }
  else
  {
    auto items = cuda::std::array<T, ItemsPerThread>{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      items[item] = input[tid * ItemsPerThread + item];
    }

    if constexpr (Mode == WarpReduceBroadcastMode::SumMultipleItems)
    {
      output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Sum(items);
    }
    else if constexpr (Mode == WarpReduceBroadcastMode::MaxMultipleItems)
    {
      output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Max(items);
    }
    else if constexpr (Mode == WarpReduceBroadcastMode::MinMultipleItems)
    {
      output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Min(items);
    }
    else
    {
      output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Reduce(items, reduction_op);
    }
  }
}

template <typename T>
__global__ void warp_reduce_broadcast_default_logical_warp_kernel(const T* input, T* output)
{
  using warp_reduce_t = cub::WarpReduceBroadcast<T>;

  __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / warp_size];

  const int tid             = static_cast<int>(threadIdx.x);
  const int logical_warp_id = tid / warp_size;

  output[tid] = warp_reduce_t{temp_storage[logical_warp_id]}.Sum(input[tid]);
}

template <typename OutputT, typename InputT, int LogicalWarpThreads, int ItemsPerThread>
__global__ void warp_reduce_broadcast_mixed_items_kernel(const InputT* input, OutputT* output)
{
  using warp_reduce_t = cub::WarpReduceBroadcast<OutputT, LogicalWarpThreads>;

  __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

  const int tid                   = static_cast<int>(threadIdx.x);
  const int block_logical_warp_id = tid / LogicalWarpThreads;

  auto items = cuda::std::array<InputT, ItemsPerThread>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    items[item] = input[tid * ItemsPerThread + item];
  }

  output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Sum(items);
}

template <typename OutputT, typename InputT, int LogicalWarpThreads, int ItemsPerThread, typename ReductionOp>
__global__ void
warp_reduce_broadcast_mixed_reduce_items_kernel(const InputT* input, OutputT* output, ReductionOp reduction_op)
{
  using warp_reduce_t = cub::WarpReduceBroadcast<OutputT, LogicalWarpThreads>;

  __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

  const int tid                   = static_cast<int>(threadIdx.x);
  const int block_logical_warp_id = tid / LogicalWarpThreads;

  auto items = cuda::std::array<InputT, ItemsPerThread>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    items[item] = input[tid * ItemsPerThread + item];
  }

  output[tid] = warp_reduce_t{temp_storage[block_logical_warp_id]}.Reduce(items, reduction_op);
}

template <typename OutputT, typename InputT, int LogicalWarpThreads, int ItemsPerThread>
__global__ void
warp_reduce_broadcast_mixed_min_max_items_kernel(const InputT* input, OutputT* max_output, OutputT* min_output)
{
  using warp_reduce_t = cub::WarpReduceBroadcast<OutputT, LogicalWarpThreads>;

  __shared__ typename warp_reduce_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

  const int tid                   = static_cast<int>(threadIdx.x);
  const int block_logical_warp_id = tid / LogicalWarpThreads;

  auto items = cuda::std::array<InputT, ItemsPerThread>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < ItemsPerThread; ++item)
  {
    items[item] = input[tid * ItemsPerThread + item];
  }

  auto warp_reduce = warp_reduce_t{temp_storage[block_logical_warp_id]};
  max_output[tid]  = warp_reduce.Max(items);
  min_output[tid]  = warp_reduce.Min(items);
}

template <typename T, int MaxReductionItems, typename ReductionOp>
void gen_bounded_input(c2h::seed_t seed, c2h::device_vector<T>& input)
{
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    if constexpr (cub::detail::is_cuda_std_plus_v<ReductionOp, T>)
    {
      c2h::gen(seed, input, T(0.5), T(1.5));
    }
    else
    {
      c2h::gen(seed, input, T(-1.5), T(1.5));
    }
  }
  else if constexpr (cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ReductionOp, T>
                     && cuda::std::is_signed_v<T>)
  {
    using bound_t           = long long;
    constexpr auto raw_max  = static_cast<bound_t>(cuda::std::numeric_limits<T>::max()) / (MaxReductionItems + 1);
    constexpr bool is_bound = raw_max > bound_t{0};
    if constexpr (is_bound)
    {
      const T gen_max = static_cast<T>(raw_max);
      const T gen_min = static_cast<T>(-gen_max);
      c2h::gen(seed, input, gen_min, gen_max);
    }
    else
    {
      c2h::host_vector<T> h_input(input.size());
      const auto seed_value = seed.get();
      const int seed_offset = static_cast<int>(seed_value % MaxReductionItems);
      const int seed_sign   = (seed_value / MaxReductionItems) % 2 == 0 ? 1 : -1;
      for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
      {
        const auto signal = ((idx / MaxReductionItems) % 2 == 0) ? T(seed_sign) : T(-seed_sign);
        h_input[idx]      = (idx % MaxReductionItems == seed_offset) ? signal : T(0);
      }
      input = h_input;
    }
  }
  else if constexpr (cuda::std::is_integral_v<T> && cub::detail::is_cuda_std_plus_v<ReductionOp, T>)
  {
    using bound_t      = unsigned long long;
    const auto raw_max = static_cast<bound_t>(cuda::std::numeric_limits<T>::max()) / (MaxReductionItems + 1);
    const T gen_max    = static_cast<T>(raw_max > bound_t{0} ? raw_max : bound_t{1});
    c2h::gen(seed, input, T(0), gen_max);
  }
  else
  {
    c2h::gen(seed, input);
  }
}

template <int LogicalWarpThreads, int ItemsPerThread, typename T, typename ReductionOp>
void compute_host_reference(
  const c2h::host_vector<T>& input, c2h::host_vector<T>& reference, ReductionOp reduction_op, int valid_items)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;
  using accumulator_t             = cuda::std::__accumulator_t<ReductionOp, T, T>;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    auto aggregate = static_cast<accumulator_t>(identity_v<ReductionOp, T>);
    for (int lane = 0; lane < valid_items; ++lane)
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const int input_idx = (logical_warp * LogicalWarpThreads + lane) * ItemsPerThread + item;
        aggregate           = reduction_op(aggregate, static_cast<accumulator_t>(input[input_idx]));
      }
    }

    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      reference[logical_warp * LogicalWarpThreads + lane] = static_cast<T>(aggregate);
    }
  }
}

template <int LogicalWarpThreads, int ItemsPerThread, typename OutputT, typename InputT>
void compute_mixed_sum_reference(const c2h::host_vector<InputT>& input, c2h::host_vector<OutputT>& reference)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    auto aggregate = OutputT{};
    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        const int input_idx = (logical_warp * LogicalWarpThreads + lane) * ItemsPerThread + item;
        aggregate += static_cast<OutputT>(input[input_idx]);
      }
    }

    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      reference[logical_warp * LogicalWarpThreads + lane] = aggregate;
    }
  }
}

template <int LogicalWarpThreads, int ItemsPerThread, typename OutputT, typename InputT>
void compute_mixed_sum_lane_conversion_reference(const c2h::host_vector<InputT>& input,
                                                 c2h::host_vector<OutputT>& reference)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;
  using accumulator_t             = cuda::std::__accumulator_t<cuda::std::plus<>, InputT, InputT>;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    auto aggregate = OutputT{};
    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      const int lane_base = (logical_warp * LogicalWarpThreads + lane) * ItemsPerThread;
      auto lane_sum       = static_cast<accumulator_t>(input[lane_base]);
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        lane_sum = cuda::std::plus<>{}(lane_sum, static_cast<accumulator_t>(input[lane_base + item]));
      }

      aggregate += static_cast<OutputT>(lane_sum);
    }

    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      reference[logical_warp * LogicalWarpThreads + lane] = aggregate;
    }
  }
}

template <int LogicalWarpThreads, int ItemsPerThread, typename OutputT, typename InputT>
void compute_mixed_min_max_reference(const c2h::host_vector<InputT>& input,
                                     c2h::host_vector<OutputT>& max_reference,
                                     c2h::host_vector<OutputT>& min_reference)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    OutputT max_aggregate{};
    OutputT min_aggregate{};
    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      const int lane_base = (logical_warp * LogicalWarpThreads + lane) * ItemsPerThread;
      auto lane_max       = input[lane_base];
      auto lane_min       = input[lane_base];
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        const auto value = input[lane_base + item];
        lane_max         = cuda::maximum<>{}(lane_max, value);
        lane_min         = cuda::minimum<>{}(lane_min, value);
      }

      const auto converted_lane_max = static_cast<OutputT>(lane_max);
      const auto converted_lane_min = static_cast<OutputT>(lane_min);
      if (lane == 0)
      {
        max_aggregate = converted_lane_max;
        min_aggregate = converted_lane_min;
      }
      else
      {
        max_aggregate = cuda::maximum<>{}(max_aggregate, converted_lane_max);
        min_aggregate = cuda::minimum<>{}(min_aggregate, converted_lane_min);
      }
    }

    for (int lane = 0; lane < LogicalWarpThreads; ++lane)
    {
      max_reference[logical_warp * LogicalWarpThreads + lane] = max_aggregate;
      min_reference[logical_warp * LogicalWarpThreads + lane] = min_aggregate;
    }
  }
}

template <int LogicalWarpThreads, bool IsPartial, typename F>
void for_each_valid_items(F callback)
{
  if constexpr (IsPartial && LogicalWarpThreads > 1)
  {
    callback(1);
    if constexpr (LogicalWarpThreads > 2)
    {
      callback(LogicalWarpThreads / 2);
      callback(LogicalWarpThreads - 1);
    }
    callback(LogicalWarpThreads);
  }
  else
  {
    callback(LogicalWarpThreads);
  }
}

template <int LogicalWarpThreads, typename T>
void verify_broadcast_output(const c2h::host_vector<T>& output)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    const int warp_begin = logical_warp * LogicalWarpThreads;
    for (int lane = 1; lane < LogicalWarpThreads; ++lane)
    {
      REQUIRE(output[warp_begin + lane] == output[warp_begin]);
    }
  }
}

template <WarpReduceBroadcastMode Mode, int LogicalWarpThreads, typename T, typename ReductionOp>
void test_warp_reduce_broadcast(ReductionOp reduction_op = ReductionOp{})
{
  constexpr int items_per_thread = items_per_thread_for_mode_v<Mode>;

  for_each_valid_items<LogicalWarpThreads, is_partial_mode_v<Mode>>([&](int valid_items) {
    CAPTURE(c2h::type_name<T>(), c2h::type_name<ReductionOp>(), LogicalWarpThreads, items_per_thread, valid_items);

    c2h::device_vector<T> d_input(block_size * items_per_thread);
    gen_bounded_input<T, LogicalWarpThreads * items_per_thread_for_mode_v<Mode>, ReductionOp>(C2H_SEED(10), d_input);

    c2h::device_vector<T> d_output(block_size);
    warp_reduce_broadcast_kernel<Mode, LogicalWarpThreads, items_per_thread_for_mode_v<Mode>><<<1, block_size>>>(
      thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), reduction_op, valid_items);

    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    c2h::host_vector<T> h_input  = d_input;
    c2h::host_vector<T> h_output = d_output;
    c2h::host_vector<T> h_reference(block_size);

    verify_broadcast_output<LogicalWarpThreads>(h_output);
    compute_host_reference<LogicalWarpThreads, items_per_thread_for_mode_v<Mode>>(
      h_input, h_reference, reduction_op, valid_items);
    verify_results(h_reference, h_output);
  });
}

namespace
{
affine_value_t make_affine_value(int idx)
{
  return affine_value_t{idx % 3 + 1, (idx * 7 + 5) % 17};
}

int make_packed_affine_value(int idx)
{
  return pack_affine_value(idx % 3 + 1, (idx * 7 + 5) % 17);
}
} // namespace

template <WarpReduceBroadcastMode Mode, int LogicalWarpThreads>
void test_warp_reduce_broadcast_non_commutative()
{
  static_assert(Mode == WarpReduceBroadcastMode::Reduce || Mode == WarpReduceBroadcastMode::ReduceValidItems
                  || Mode == WarpReduceBroadcastMode::ReduceMultipleItems,
                "Only Reduce modes accept the non-commutative operator");

  constexpr int items_per_thread = items_per_thread_for_mode_v<Mode>;

  for_each_valid_items<LogicalWarpThreads, is_partial_mode_v<Mode>>([&](int valid_items) {
    c2h::host_vector<affine_value_t> h_input(block_size * items_per_thread);
    for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
    {
      h_input[idx] = make_affine_value(idx);
    }

    c2h::device_vector<affine_value_t> d_input = h_input;
    c2h::device_vector<affine_value_t> d_output(block_size);
    warp_reduce_broadcast_kernel<Mode, LogicalWarpThreads, items_per_thread_for_mode_v<Mode>><<<1, block_size>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()),
      affine_compose_op{},
      valid_items);

    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    c2h::host_vector<affine_value_t> h_output = d_output;
    c2h::host_vector<affine_value_t> h_reference(block_size);

    verify_broadcast_output<LogicalWarpThreads>(h_output);
    // This compares the broadcast wrapper against the same operand ordering used by WarpReduce::Reduce for
    // associative, non-commutative operators.
    compute_host_reference<LogicalWarpThreads, items_per_thread_for_mode_v<Mode>>(
      h_input, h_reference, affine_compose_op{}, valid_items);
    verify_results(h_reference, h_output);
  });
}

template <WarpReduceBroadcastMode Mode, int LogicalWarpThreads>
void test_warp_reduce_broadcast_integral_non_commutative()
{
  static_assert(Mode == WarpReduceBroadcastMode::Reduce || Mode == WarpReduceBroadcastMode::ReduceValidItems
                  || Mode == WarpReduceBroadcastMode::ReduceMultipleItems,
                "Only Reduce modes accept the non-commutative operator");

  constexpr int items_per_thread = items_per_thread_for_mode_v<Mode>;

  for_each_valid_items<LogicalWarpThreads, is_partial_mode_v<Mode>>([&](int valid_items) {
    c2h::host_vector<int> h_input(block_size * items_per_thread);
    for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
    {
      h_input[idx] = make_packed_affine_value(idx);
    }

    c2h::device_vector<int> d_input = h_input;
    c2h::device_vector<int> d_output(block_size);
    warp_reduce_broadcast_kernel<Mode, LogicalWarpThreads, items_per_thread_for_mode_v<Mode>><<<1, block_size>>>(
      thrust::raw_pointer_cast(d_input.data()),
      thrust::raw_pointer_cast(d_output.data()),
      packed_affine_compose_op{},
      valid_items);

    REQUIRE(cudaSuccess == cudaPeekAtLastError());
    REQUIRE(cudaSuccess == cudaDeviceSynchronize());

    c2h::host_vector<int> h_output = d_output;
    c2h::host_vector<int> h_reference(block_size);

    verify_broadcast_output<LogicalWarpThreads>(h_output);
    compute_host_reference<LogicalWarpThreads, items_per_thread_for_mode_v<Mode>>(
      h_input, h_reference, packed_affine_compose_op{}, valid_items);
    verify_results(h_reference, h_output);
  });
}

template <typename T>
void test_warp_reduce_broadcast_small_sum_signal()
{
  constexpr int logical_warp_num_threads = 16;
  constexpr int items_per_thread         = 4;

  c2h::host_vector<T> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
  {
    if constexpr (cuda::std::is_signed_v<T>)
    {
      h_input[idx] = ((idx % 4) == 0) ? T(-1) : T(0);
    }
    else
    {
      h_input[idx] = ((idx % 3) == 0) ? T(1) : T(0);
    }
  }

  c2h::device_vector<T> d_input = h_input;
  c2h::device_vector<T> d_output(block_size);
  warp_reduce_broadcast_kernel<WarpReduceBroadcastMode::SumMultipleItems, logical_warp_num_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                        thrust::raw_pointer_cast(d_output.data()),
                        cuda::std::plus<>{},
                        logical_warp_num_threads);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<T> h_output = d_output;
  c2h::host_vector<T> h_reference(block_size);

  verify_broadcast_output<logical_warp_num_threads>(h_output);
  compute_host_reference<logical_warp_num_threads, items_per_thread>(
    h_input, h_reference, cuda::std::plus<>{}, logical_warp_num_threads);
  verify_results(h_reference, h_output);
}

using value_types =
  c2h::type_list<cuda::std::int8_t,
                 cuda::std::uint8_t,
                 cuda::std::int16_t,
                 cuda::std::uint16_t,
                 cuda::std::int32_t,
                 cuda::std::uint32_t,
                 cuda::std::int64_t,
                 cuda::std::uint64_t,
                 float,
                 double>;
using reduction_ops = c2h::type_list<cuda::std::plus<>, cuda::maximum<>, cuda::minimum<>>;
using bitwise_value_types =
  c2h::type_list<cuda::std::int8_t,
                 cuda::std::uint8_t,
                 cuda::std::int16_t,
                 cuda::std::uint16_t,
                 cuda::std::int32_t,
                 cuda::std::uint32_t,
                 cuda::std::int64_t,
                 cuda::std::uint64_t>;
using bitwise_reduction_ops = c2h::type_list<cuda::std::bit_and<>, cuda::std::bit_or<>, cuda::std::bit_xor<>>;
using logical_warp_threads  = c2h::enum_type_list<int, 32, 16, 8, 4, 2, 1>;

C2H_TEST("WarpReduceBroadcast uses the default logical warp size", "[warp][reduce][broadcast][default]")
{
  c2h::device_vector<int> d_input(block_size);
  c2h::gen(C2H_SEED(10), d_input, 0, 7);

  c2h::device_vector<int> d_output(block_size);
  warp_reduce_broadcast_default_logical_warp_kernel<<<1, block_size>>>(
    thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_input  = d_input;
  c2h::host_vector<int> h_output = d_output;
  c2h::host_vector<int> h_reference(block_size);

  verify_broadcast_output<warp_size>(h_output);
  compute_host_reference<warp_size, 1>(h_input, h_reference, cuda::std::plus<>{}, warp_size);
  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBroadcast::Sum and Reduce support multiple items with a wider accumulator",
         "[warp][reduce][broadcast][sum][items][mixed]")
{
  using output_t                 = cuda::std::int64_t;
  using input_t                  = cuda::std::int32_t;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (auto& value : h_input)
  {
    value = input_t{100000000};
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_sum_output(block_size);
  c2h::device_vector<output_t> d_reduce_output(block_size);
  warp_reduce_broadcast_mixed_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_sum_output.data()));
  warp_reduce_broadcast_mixed_reduce_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(
      thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_reduce_output.data()), cuda::std::plus<>{});

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_sum_output    = d_sum_output;
  c2h::host_vector<output_t> h_reduce_output = d_reduce_output;
  c2h::host_vector<output_t> h_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_sum_output);
  verify_broadcast_output<mixed_logical_warp_threads>(h_reduce_output);
  compute_mixed_sum_reference<mixed_logical_warp_threads, items_per_thread>(h_input, h_reference);
  verify_results(h_reference, h_sum_output);
  verify_results(h_reference, h_reduce_output);
}

C2H_TEST("WarpReduceBroadcast::Sum of multiple items converts signed lanes to unsigned output",
         "[warp][reduce][broadcast][sum][items][mixed][signedness]")
{
  using output_t                 = cuda::std::uint8_t;
  using input_t                  = cuda::std::int32_t;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); idx += items_per_thread)
  {
    h_input[idx + 0] = input_t{-3};
    h_input[idx + 1] = input_t{1};
    h_input[idx + 2] = input_t{0};
    h_input[idx + 3] = input_t{0};
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_output(block_size);
  warp_reduce_broadcast_mixed_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_output = d_output;
  c2h::host_vector<output_t> h_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_output);
  compute_mixed_sum_lane_conversion_reference<mixed_logical_warp_threads, items_per_thread>(h_input, h_reference);
  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBroadcast::Sum of multiple items converts unsigned lanes to signed output",
         "[warp][reduce][broadcast][sum][items][mixed][signedness]")
{
  using output_t                 = cuda::std::int16_t;
  using input_t                  = cuda::std::uint32_t;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); idx += items_per_thread)
  {
    h_input[idx + 0] = input_t{1000};
    h_input[idx + 1] = input_t{23};
    h_input[idx + 2] = input_t{0};
    h_input[idx + 3] = input_t{0};
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_output(block_size);
  warp_reduce_broadcast_mixed_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_output = d_output;
  c2h::host_vector<output_t> h_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_output);
  compute_mixed_sum_lane_conversion_reference<mixed_logical_warp_threads, items_per_thread>(h_input, h_reference);
  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBroadcast::Sum of multiple non-integral items converts each lane before all-reduce",
         "[warp][reduce][broadcast][sum][items][mixed][conversion]")
{
  using output_t                 = cuda::std::int32_t;
  using input_t                  = float;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); idx += items_per_thread)
  {
    h_input[idx + 0] = input_t{0.25f};
    h_input[idx + 1] = input_t{0.50f};
    h_input[idx + 2] = input_t{0.50f};
    h_input[idx + 3] = input_t{0.50f};
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_output(block_size);
  warp_reduce_broadcast_mixed_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_output = d_output;
  c2h::host_vector<output_t> h_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_output);
  compute_mixed_sum_lane_conversion_reference<mixed_logical_warp_threads, items_per_thread>(h_input, h_reference);
  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBroadcast::Max and Min match WarpReduce mixed item conversion",
         "[warp][reduce][broadcast][min][max][items][mixed]")
{
  using output_t                 = cuda::std::int32_t;
  using input_t                  = cuda::std::uint32_t;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); ++idx)
  {
    const auto item = idx % items_per_thread;
    if (item == 0)
    {
      h_input[idx] = static_cast<input_t>(cuda::std::numeric_limits<output_t>::min());
    }
    else
    {
      h_input[idx] = static_cast<input_t>(item);
    }
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_max_output(block_size);
  c2h::device_vector<output_t> d_min_output(block_size);
  warp_reduce_broadcast_mixed_min_max_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                        thrust::raw_pointer_cast(d_max_output.data()),
                        thrust::raw_pointer_cast(d_min_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_max_output = d_max_output;
  c2h::host_vector<output_t> h_min_output = d_min_output;
  c2h::host_vector<output_t> h_max_reference(block_size);
  c2h::host_vector<output_t> h_min_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_max_output);
  verify_broadcast_output<mixed_logical_warp_threads>(h_min_output);
  compute_mixed_min_max_reference<mixed_logical_warp_threads, items_per_thread>(
    h_input, h_max_reference, h_min_reference);
  verify_results(h_max_reference, h_max_output);
  verify_results(h_min_reference, h_min_output);
}

C2H_TEST("WarpReduceBroadcast::Max and Min convert after each lane's item reduction",
         "[warp][reduce][broadcast][min][max][items][mixed][narrow]")
{
  using output_t                 = cuda::std::uint8_t;
  using input_t                  = cuda::std::uint32_t;
  constexpr int items_per_thread = 4;

  c2h::host_vector<input_t> h_input(block_size * items_per_thread);
  for (int idx = 0; idx < static_cast<int>(h_input.size()); idx += items_per_thread)
  {
    h_input[idx + 0] = input_t{300};
    h_input[idx + 1] = input_t{255};
    h_input[idx + 2] = input_t{256};
    h_input[idx + 3] = input_t{1};
  }

  c2h::device_vector<input_t> d_input = h_input;
  c2h::device_vector<output_t> d_max_output(block_size);
  c2h::device_vector<output_t> d_min_output(block_size);
  warp_reduce_broadcast_mixed_min_max_items_kernel<output_t, input_t, mixed_logical_warp_threads, items_per_thread>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                        thrust::raw_pointer_cast(d_max_output.data()),
                        thrust::raw_pointer_cast(d_min_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<output_t> h_max_output = d_max_output;
  c2h::host_vector<output_t> h_min_output = d_min_output;
  c2h::host_vector<output_t> h_max_reference(block_size);
  c2h::host_vector<output_t> h_min_reference(block_size);

  verify_broadcast_output<mixed_logical_warp_threads>(h_max_output);
  verify_broadcast_output<mixed_logical_warp_threads>(h_min_output);
  compute_mixed_min_max_reference<mixed_logical_warp_threads, items_per_thread>(
    h_input, h_max_reference, h_min_reference);
  verify_results(h_max_reference, h_max_output);
  verify_results(h_min_reference, h_min_output);
}

C2H_TEST("WarpReduceBroadcast::Sum of 8-bit multiple items produces non-zero aggregates",
         "[warp][reduce][broadcast][sum][items][small]")
{
  test_warp_reduce_broadcast_small_sum_signal<cuda::std::int8_t>();
  test_warp_reduce_broadcast_small_sum_signal<cuda::std::uint8_t>();
}

C2H_TEST("WarpReduceBroadcast::Sum returns the aggregate to every lane",
         "[warp][reduce][broadcast][sum]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::Sum, logical_warp_t::value, value_t, cuda::std::plus<>>();
}

C2H_TEST("WarpReduceBroadcast::Max returns the aggregate to every lane",
         "[warp][reduce][broadcast][max]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::Max, logical_warp_t::value, value_t, cuda::maximum<>>();
}

C2H_TEST("WarpReduceBroadcast::Min returns the aggregate to every lane",
         "[warp][reduce][broadcast][min]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::Min, logical_warp_t::value, value_t, cuda::minimum<>>();
}

C2H_TEST("WarpReduceBroadcast::Reduce returns the aggregate to every lane",
         "[warp][reduce][broadcast]",
         value_types,
         reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::Reduce, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Reduce supports bitwise operators",
         "[warp][reduce][broadcast][bitwise]",
         bitwise_value_types,
         bitwise_reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::Reduce, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Sum with valid items returns the aggregate to every lane",
         "[warp][reduce][broadcast][sum][partial]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::SumValidItems, logical_warp_t::value, value_t, cuda::std::plus<>>();
}

C2H_TEST("WarpReduceBroadcast::Max with valid items returns the aggregate to every lane",
         "[warp][reduce][broadcast][max][partial]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::MaxValidItems, logical_warp_t::value, value_t, cuda::maximum<>>();
}

C2H_TEST("WarpReduceBroadcast::Min with valid items returns the aggregate to every lane",
         "[warp][reduce][broadcast][min][partial]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::MinValidItems, logical_warp_t::value, value_t, cuda::minimum<>>();
}

C2H_TEST("WarpReduceBroadcast::Reduce with valid items returns the aggregate to every lane",
         "[warp][reduce][broadcast][partial]",
         value_types,
         reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::ReduceValidItems, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Sum of multiple items returns the aggregate to every lane",
         "[warp][reduce][broadcast][sum][items]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::SumMultipleItems,
                             logical_warp_t::value,
                             value_t,
                             cuda::std::plus<>>();
}

C2H_TEST("WarpReduceBroadcast::Max of multiple items returns the aggregate to every lane",
         "[warp][reduce][broadcast][max][items]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::MaxMultipleItems, logical_warp_t::value, value_t, cuda::maximum<>>();
}

C2H_TEST("WarpReduceBroadcast::Min of multiple items returns the aggregate to every lane",
         "[warp][reduce][broadcast][min][items]",
         value_types,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using logical_warp_t = c2h::get<1, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::MinMultipleItems, logical_warp_t::value, value_t, cuda::minimum<>>();
}

C2H_TEST("WarpReduceBroadcast::Reduce of multiple items returns the aggregate to every lane",
         "[warp][reduce][broadcast][items]",
         value_types,
         reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::ReduceMultipleItems, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Reduce of multiple items supports bitwise operators",
         "[warp][reduce][broadcast][items][bitwise]",
         bitwise_value_types,
         bitwise_reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::ReduceMultipleItems, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Reduce with valid items supports bitwise operators",
         "[warp][reduce][broadcast][partial][bitwise]",
         bitwise_value_types,
         bitwise_reduction_ops,
         logical_warp_threads)
{
  using value_t        = c2h::get<0, TestType>;
  using op_t           = c2h::get<1, TestType>;
  using logical_warp_t = c2h::get<2, TestType>;

  test_warp_reduce_broadcast<WarpReduceBroadcastMode::ReduceValidItems, logical_warp_t::value, value_t, op_t>();
}

C2H_TEST("WarpReduceBroadcast::Reduce preserves non-commutative operand order",
         "[warp][reduce][broadcast][non_commutative]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_non_commutative<WarpReduceBroadcastMode::Reduce, logical_warp_t::value>();
}

C2H_TEST("WarpReduceBroadcast::Reduce with valid items preserves non-commutative operand order",
         "[warp][reduce][broadcast][partial][non_commutative]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_non_commutative<WarpReduceBroadcastMode::ReduceValidItems, logical_warp_t::value>();
}

C2H_TEST("WarpReduceBroadcast::Reduce of multiple items preserves non-commutative operand order",
         "[warp][reduce][broadcast][items][non_commutative]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_non_commutative<WarpReduceBroadcastMode::ReduceMultipleItems, logical_warp_t::value>();
}

C2H_TEST("WarpReduceBroadcast::Reduce preserves integral non-commutative operand order",
         "[warp][reduce][broadcast][non_commutative][integral]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_integral_non_commutative<WarpReduceBroadcastMode::Reduce, logical_warp_t::value>();
}

C2H_TEST("WarpReduceBroadcast::Reduce with valid items preserves integral non-commutative operand order",
         "[warp][reduce][broadcast][partial][non_commutative][integral]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_integral_non_commutative<WarpReduceBroadcastMode::ReduceValidItems, logical_warp_t::value>();
}

C2H_TEST("WarpReduceBroadcast::Reduce of multiple items preserves integral non-commutative operand order",
         "[warp][reduce][broadcast][items][non_commutative][integral]",
         logical_warp_threads)
{
  using logical_warp_t = c2h::get<0, TestType>;

  test_warp_reduce_broadcast_integral_non_commutative<WarpReduceBroadcastMode::ReduceMultipleItems,
                                                      logical_warp_t::value>();
}
