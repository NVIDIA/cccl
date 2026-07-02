// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_reduce_batched_broadcast.cuh>

#include <cuda/functional>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <test_util.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/custom_type.h>
#include <c2h/operator.cuh>

inline constexpr int warp_size  = 32;
inline constexpr int block_size = 2 * warp_size;

enum class WarpReduceBatchedBroadcastMode
{
  Sum,
  SumOutput,
  CommutativeReduce,
  CommutativeReduceOutput
};

using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

template <typename T>
inline constexpr bool is_ulonglong4_test_type_v =
#if _CCCL_CTK_AT_LEAST(13, 0)
  ::cuda::std::is_same_v<T, ulonglong4_16a>;
#else // _CCCL_CTK_AT_LEAST(13, 0)
  ::cuda::std::is_same_v<T, ulonglong4>;
#endif // _CCCL_CTK_AT_LEAST(13, 0)

template <typename T>
inline constexpr bool has_unwritten_output_sentinel_v =
  ::cuda::std::is_floating_point_v<T> || ::cuda::std::is_integral_v<T>;

template <typename T>
_CCCL_HOST_DEVICE T unwritten_output_sentinel()
{
  static_assert(has_unwritten_output_sentinel_v<T>, "The output sentinel is only defined for arithmetic test types");
  if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    return T{-123.25};
  }
  else
  {
    return ::cuda::std::numeric_limits<T>::max();
  }
}

template <WarpReduceBatchedBroadcastMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
__global__ void warp_reduce_batched_broadcast_kernel(const T* input, T* output, ReductionOp reduction_op)
{
  using warp_reduce_t = cub::WarpReduceBatchedBroadcast<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;

  typename warp_reduce_t::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);

  auto inputs = ::cuda::std::array<T, Batches>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < Batches; ++batch)
  {
    inputs[batch] = input[tid * Batches + batch];
  }

  auto outputs = ::cuda::std::array<T, Batches>{};
  if constexpr (Mode == WarpReduceBatchedBroadcastMode::SumOutput
                || Mode == WarpReduceBatchedBroadcastMode::CommutativeReduceOutput)
  {
    if constexpr (has_unwritten_output_sentinel_v<T>)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int batch = 0; batch < Batches; ++batch)
      {
        outputs[batch] = unwritten_output_sentinel<T>();
      }
    }
  }

  if constexpr (Mode == WarpReduceBatchedBroadcastMode::Sum)
  {
    outputs = warp_reduce_t{temp_storage}.Sum(inputs);
  }
  else if constexpr (Mode == WarpReduceBatchedBroadcastMode::SumOutput)
  {
    warp_reduce_t{temp_storage}.Sum(inputs, outputs);
  }
  else if constexpr (Mode == WarpReduceBatchedBroadcastMode::CommutativeReduce)
  {
    outputs = warp_reduce_t{temp_storage}.CommutativeReduce(inputs, reduction_op);
  }
  else
  {
    warp_reduce_t{temp_storage}.CommutativeReduce(inputs, outputs, reduction_op);
  }

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < Batches; ++batch)
  {
    output[tid * Batches + batch] = outputs[batch];
  }
}

template <WarpReduceBatchedBroadcastMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
__global__ void warp_reduce_batched_broadcast_cond_part_kernel(const T* input, T* output, ReductionOp reduction_op)
{
  static_assert(LogicalWarpThreads < warp_size, "Need at least 2 logical warps per physical warp for this test");
  static_assert(!SyncPhysicalWarp, "Divergent participation is only valid without physical-warp synchronization");
  using warp_reduce_t = cub::WarpReduceBatchedBroadcast<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;

  typename warp_reduce_t::TempStorage temp_storage;

  const int tid               = static_cast<int>(threadIdx.x);
  const int logical_warp_id   = tid / LogicalWarpThreads;
  const bool is_participating = logical_warp_id % 2 == 0;

  if (is_participating)
  {
    auto inputs = ::cuda::std::array<T, Batches>{};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      inputs[batch] = input[tid * Batches + batch];
    }

    auto outputs = ::cuda::std::array<T, Batches>{};
    if constexpr (Mode == WarpReduceBatchedBroadcastMode::SumOutput
                  || Mode == WarpReduceBatchedBroadcastMode::CommutativeReduceOutput)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int batch = 0; batch < Batches; ++batch)
      {
        outputs[batch] = unwritten_output_sentinel<T>();
      }
    }

    if constexpr (Mode == WarpReduceBatchedBroadcastMode::Sum)
    {
      outputs = warp_reduce_t{temp_storage}.Sum(inputs);
    }
    else if constexpr (Mode == WarpReduceBatchedBroadcastMode::SumOutput)
    {
      warp_reduce_t{temp_storage}.Sum(inputs, outputs);
    }
    else if constexpr (Mode == WarpReduceBatchedBroadcastMode::CommutativeReduce)
    {
      outputs = warp_reduce_t{temp_storage}.CommutativeReduce(inputs, reduction_op);
    }
    else
    {
      warp_reduce_t{temp_storage}.CommutativeReduce(inputs, outputs, reduction_op);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int batch = 0; batch < Batches; ++batch)
    {
      output[tid * Batches + batch] = outputs[batch];
    }
  }

  if constexpr (!SyncPhysicalWarp)
  {
    // Re-converge the full physical warp after the divergent region to surface unexpected deadlocks.
    __syncwarp();
  }
}

template <typename T, int Batches>
__global__ void warp_reduce_batched_broadcast_default_logical_warp_kernel(const T* input, T* output)
{
  using warp_reduce_t = cub::WarpReduceBatchedBroadcast<T, Batches>;

  typename warp_reduce_t::TempStorage temp_storage;

  const int tid = static_cast<int>(threadIdx.x);

  auto inputs = ::cuda::std::array<T, Batches>{};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < Batches; ++batch)
  {
    inputs[batch] = input[tid * Batches + batch];
  }

  const auto outputs = warp_reduce_t{temp_storage}.Sum(inputs);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int batch = 0; batch < Batches; ++batch)
  {
    output[tid * Batches + batch] = outputs[batch];
  }
}

template <typename T, int MaxReductionItems>
void gen_bounded_input(c2h::seed_t seed, c2h::device_vector<T>& input)
{
  if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    c2h::gen(seed, input, T(0.5), T(1.5));
  }
  else if constexpr (::cuda::std::is_integral_v<T> && ::cuda::std::is_signed_v<T>)
  {
    const T gen_max = ::cuda::std::numeric_limits<T>::max() / static_cast<T>(MaxReductionItems + 1);
    c2h::gen(seed, input, -gen_max, gen_max);
  }
  else if constexpr (::cuda::std::is_integral_v<T>)
  {
    const T gen_max = ::cuda::std::numeric_limits<T>::max() / static_cast<T>(MaxReductionItems + 1);
    c2h::gen(seed, input, T(0), gen_max);
  }
  else if constexpr (is_ulonglong4_test_type_v<T>)
  {
    using component_t             = decltype(T::x);
    const component_t gen_max     = ::cuda::std::numeric_limits<component_t>::max() / MaxReductionItems;
    constexpr component_t gen_min = 0;
    c2h::gen(seed, input, T{gen_min, gen_min, gen_min, gen_min}, T{gen_max, gen_max, gen_max, gen_max});
  }
  else if constexpr (::cuda::std::is_same_v<T, custom_t>)
  {
    T min{};
    T max{};
    max.key = 64 / MaxReductionItems;
    max.val = 96 / MaxReductionItems;
    c2h::gen(seed, input, min, max);
  }
  else
  {
    c2h::gen(seed, input);
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // numeric(33): C: '=': conversion from 'int' to '_Ty', possible loss of data

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
void compute_host_reference(const c2h::host_vector<T>& input, c2h::host_vector<T>& reference, ReductionOp reduction_op)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;

  for (int logical_warp = 0; logical_warp < num_logical_warps; ++logical_warp)
  {
    for (int batch = 0; batch < Batches; ++batch)
    {
      auto aggregate = identity_v<ReductionOp, T>;
      for (int lane = 0; lane < LogicalWarpThreads; ++lane)
      {
        aggregate = reduction_op(aggregate, input[(logical_warp * LogicalWarpThreads + lane) * Batches + batch]);
      }

      for (int lane = 0; lane < LogicalWarpThreads; ++lane)
      {
        reference[(logical_warp * LogicalWarpThreads + lane) * Batches + batch] = static_cast<T>(aggregate);
      }
    }
  }
}

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
void compute_cond_part_host_reference(
  const c2h::host_vector<T>& input, c2h::host_vector<T>& reference, ReductionOp reduction_op)
{
  constexpr int num_logical_warps = block_size / LogicalWarpThreads;

  for (int logical_warp = 0; logical_warp < num_logical_warps; logical_warp += 2)
  {
    for (int batch = 0; batch < Batches; ++batch)
    {
      auto aggregate = identity_v<ReductionOp, T>;
      for (int lane = 0; lane < LogicalWarpThreads; ++lane)
      {
        aggregate = reduction_op(aggregate, input[(logical_warp * LogicalWarpThreads + lane) * Batches + batch]);
      }

      for (int lane = 0; lane < LogicalWarpThreads; ++lane)
      {
        reference[(logical_warp * LogicalWarpThreads + lane) * Batches + batch] = static_cast<T>(aggregate);
      }
    }
  }
}

_CCCL_DIAG_POP

template <WarpReduceBatchedBroadcastMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
void test_warp_reduce_batched_broadcast(ReductionOp reduction_op = ReductionOp{})
{
  CAPTURE(c2h::type_name<T>(), c2h::type_name<ReductionOp>(), Batches, LogicalWarpThreads, SyncPhysicalWarp);

  c2h::device_vector<T> d_input(block_size * Batches);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(block_size * Batches);
  warp_reduce_batched_broadcast_kernel<Mode, SyncPhysicalWarp, Batches, LogicalWarpThreads><<<1, block_size>>>(
    thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), reduction_op);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;
  c2h::host_vector<T> h_reference(h_input.size());

  if constexpr (Mode == WarpReduceBatchedBroadcastMode::SumOutput
                || Mode == WarpReduceBatchedBroadcastMode::CommutativeReduceOutput)
  {
    if constexpr (has_unwritten_output_sentinel_v<T>)
    {
      const T sentinel = unwritten_output_sentinel<T>();
      for (const auto& item : h_output)
      {
        REQUIRE(item != sentinel);
      }
    }
  }

  compute_host_reference<Batches, LogicalWarpThreads>(h_input, h_reference, reduction_op);
  verify_results(h_reference, h_output);
}

template <WarpReduceBatchedBroadcastMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
void test_warp_reduce_batched_broadcast_cond_participation(ReductionOp reduction_op = ReductionOp{})
{
  CAPTURE(c2h::type_name<T>(), c2h::type_name<ReductionOp>(), Batches, LogicalWarpThreads, SyncPhysicalWarp);

  c2h::device_vector<T> d_input(block_size * Batches);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(12), d_input);

  const T sentinel = unwritten_output_sentinel<T>();
  c2h::device_vector<T> d_output(block_size * Batches, sentinel);
  warp_reduce_batched_broadcast_cond_part_kernel<Mode, SyncPhysicalWarp, Batches, LogicalWarpThreads>
    <<<1, block_size>>>(
      thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), reduction_op);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;
  c2h::host_vector<T> h_reference(block_size * Batches, sentinel);

  compute_cond_part_host_reference<Batches, LogicalWarpThreads>(h_input, h_reference, reduction_op);
  verify_results(h_reference, h_output);
}

using value_types = c2h::type_list<::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t, float>;
#if _CCCL_CTK_AT_LEAST(13, 0)
using shuffle_value_types = c2h::type_list<ulonglong4_16a, custom_t>;
#else // _CCCL_CTK_AT_LEAST(13, 0)
using shuffle_value_types = c2h::type_list<ulonglong4, custom_t>;
#endif // _CCCL_CTK_AT_LEAST(13, 0)
using logical_warp_threads = c2h::enum_type_list<int, 32, 16, 8, 4, 2, 1>;
// Divergent-participation coverage excludes full-warp groups because SyncPhysicalWarp=false is the behavior under test.
using sub_warp_threads      = c2h::enum_type_list<int, 16, 8, 4, 2, 1>;
using batch_counts          = c2h::enum_type_list<int, 1, 5>;
using cond_part_value_types = c2h::type_list<::cuda::std::int32_t, float>;
using reduction_ops         = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, ::cuda::minimum<>>;

C2H_TEST("WarpReduceBatchedBroadcast uses the default logical warp size", "[warp][reduce][batched][default]")
{
  constexpr int num_batches = 3;

  c2h::device_vector<int> d_input(block_size * num_batches);
  gen_bounded_input<int, warp_size>(C2H_SEED(10), d_input);

  c2h::device_vector<int> d_output(block_size * num_batches);
  warp_reduce_batched_broadcast_default_logical_warp_kernel<int, num_batches>
    <<<1, block_size>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()));

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> h_input     = d_input;
  c2h::host_vector<int> h_output    = d_output;
  c2h::host_vector<int> h_reference = d_output;

  compute_host_reference<num_batches, warp_size>(h_input, h_reference, ::cuda::std::plus<>{});
  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBatchedBroadcast::Sum returns every batch to every lane",
         "[warp][reduce][batched]",
         value_types,
         logical_warp_threads,
         batch_counts)
{
  using value_t                          = typename c2h::get<0, TestType>;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::value;
  constexpr int num_batches              = c2h::get<2, TestType>::value;
  constexpr bool sync_physical_warp      = false;

  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::Sum,
                                     sync_physical_warp,
                                     num_batches,
                                     logical_warp_num_threads,
                                     value_t,
                                     ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::SumOutput,
                                     sync_physical_warp,
                                     num_batches,
                                     logical_warp_num_threads,
                                     value_t,
                                     ::cuda::std::plus<>>();

  if constexpr (logical_warp_num_threads < warp_size)
  {
    constexpr bool sync_physical_warp_smoke = true;
    test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::Sum,
                                       sync_physical_warp_smoke,
                                       num_batches,
                                       logical_warp_num_threads,
                                       value_t,
                                       ::cuda::std::plus<>>();
    test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::SumOutput,
                                       sync_physical_warp_smoke,
                                       num_batches,
                                       logical_warp_num_threads,
                                       value_t,
                                       ::cuda::std::plus<>>();
  }
}

C2H_TEST("WarpReduceBatchedBroadcast::CommutativeReduce returns every batch to every lane",
         "[warp][reduce][batched]",
         value_types,
         logical_warp_threads,
         batch_counts,
         reduction_ops)
{
  using value_t                          = typename c2h::get<0, TestType>;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::value;
  constexpr int num_batches              = c2h::get<2, TestType>::value;
  using op_t                             = typename c2h::get<3, TestType>;

  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduce,
                                     false,
                                     num_batches,
                                     logical_warp_num_threads,
                                     value_t,
                                     op_t>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduceOutput,
                                     false,
                                     num_batches,
                                     logical_warp_num_threads,
                                     value_t,
                                     op_t>();

  if constexpr (logical_warp_num_threads < warp_size)
  {
    test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduce,
                                       true,
                                       num_batches,
                                       logical_warp_num_threads,
                                       value_t,
                                       op_t>();
    test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduceOutput,
                                       true,
                                       num_batches,
                                       logical_warp_num_threads,
                                       value_t,
                                       op_t>();
  }
}

C2H_TEST("WarpReduceBatchedBroadcast covers common batch counts", "[warp][reduce][batched]")
{
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::Sum,
                                     false,
                                     2,
                                     4,
                                     ::cuda::std::int32_t,
                                     ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::SumOutput,
                                     false,
                                     2,
                                     4,
                                     ::cuda::std::int32_t,
                                     ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduce,
                                     false,
                                     4,
                                     8,
                                     float,
                                     ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduceOutput,
                                     false,
                                     4,
                                     8,
                                     float,
                                     ::cuda::std::plus<>>();
}

C2H_TEST("WarpReduceBatchedBroadcast handles divergent logical-warp participation",
         "[warp][reduce][batched]",
         cond_part_value_types,
         sub_warp_threads,
         batch_counts,
         reduction_ops)
{
  using value_t                          = typename c2h::get<0, TestType>;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::value;
  constexpr int num_batches              = c2h::get<2, TestType>::value;
  using op_t                             = typename c2h::get<3, TestType>;

  test_warp_reduce_batched_broadcast_cond_participation<
    WarpReduceBatchedBroadcastMode::Sum,
    false,
    num_batches,
    logical_warp_num_threads,
    value_t,
    ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast_cond_participation<
    WarpReduceBatchedBroadcastMode::SumOutput,
    false,
    num_batches,
    logical_warp_num_threads,
    value_t,
    ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast_cond_participation<
    WarpReduceBatchedBroadcastMode::CommutativeReduce,
    false,
    num_batches,
    logical_warp_num_threads,
    value_t,
    op_t>();
  test_warp_reduce_batched_broadcast_cond_participation<
    WarpReduceBatchedBroadcastMode::CommutativeReduceOutput,
    false,
    num_batches,
    logical_warp_num_threads,
    value_t,
    op_t>();
}

C2H_TEST("WarpReduceBatchedBroadcast handles multi-word shuffled value types",
         "[warp][reduce][batched]",
         shuffle_value_types)
{
  using value_t = typename c2h::get<0, TestType>;

  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::Sum, false, 4, 4, value_t, ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::SumOutput, false, 4, 4, value_t, ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduce,
                                     false,
                                     4,
                                     4,
                                     value_t,
                                     ::cuda::std::plus<>>();
  test_warp_reduce_batched_broadcast<WarpReduceBatchedBroadcastMode::CommutativeReduceOutput,
                                     false,
                                     4,
                                     4,
                                     value_t,
                                     ::cuda::std::plus<>>();
}
