// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>

#include <cuda/cmath>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/ptx>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <numeric>

#include <test_util.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/custom_type.h>
#include <c2h/operator.cuh>

// %PARAM% TEST_TYPES types 0:1:2

inline constexpr int warp_size  = 32;
inline constexpr int block_size = 2 * warp_size;

// 2D layout: num_batches x batch_size (row-major). Single type with dynamic extents for kernels and host.
template <typename T>
using input_2d_mdspan_t = cuda::std::mdspan<T, cuda::std::dextents<int, 2>>;

enum class WarpReduceBatchedMode
{
  SingleOut,
  ToStriped,
  ToBlocked
};

template <WarpReduceBatchedMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
__global__ void
warp_reduce_batched_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output, ReductionOp reduction_op)
{
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage[block_size / LogicalWarpThreads];
  const int tid             = threadIdx.x;
  const int logical_warp_id = tid / LogicalWarpThreads;
  const int lane_id         = tid % LogicalWarpThreads;

  auto inputs = cuda::std::array<T, Batches>{};
  for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
  {
    inputs[batch_idx] = input_md(logical_warp_id * Batches + batch_idx, lane_id);
  }

  constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
  auto outputs                 = cuda::std::array<T, out_per_thread>{};
  if constexpr (Mode == WarpReduceBatchedMode::SingleOut)
  {
    outputs[0] = warp_reduce_batched_t{temp_storage[logical_warp_id]}.Reduce(inputs, reduction_op);
  }
  if constexpr (Mode == WarpReduceBatchedMode::ToBlocked)
  {
    warp_reduce_batched_t{temp_storage[logical_warp_id]}.ReduceToBlocked(inputs, outputs, reduction_op);
  }
  if constexpr (Mode == WarpReduceBatchedMode::ToStriped)
  {
    warp_reduce_batched_t{temp_storage[logical_warp_id]}.ReduceToStriped(inputs, outputs, reduction_op);
  }

  for (int idx = 0; idx < out_per_thread; ++idx)
  {
    const auto batch_idx =
      (Mode == WarpReduceBatchedMode::ToBlocked)
        ? (idx + lane_id * out_per_thread)
        : (idx * LogicalWarpThreads + lane_id);
    if (batch_idx < Batches)
    {
      output[logical_warp_id * Batches + batch_idx] = outputs[idx];
    }
  }
}

template <WarpReduceBatchedMode Mode, bool SyncPhysicalWarp, int Batches, int LogicalWarpThreads, typename T>
__global__ void sum_batched_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output)
{
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads, SyncPhysicalWarp>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage[block_size / LogicalWarpThreads];
  const int tid             = threadIdx.x;
  const int logical_warp_id = tid / LogicalWarpThreads;
  const int lane_id         = tid % LogicalWarpThreads;

  auto inputs = cuda::std::array<T, Batches>{};
  for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
  {
    inputs[batch_idx] = input_md(logical_warp_id * Batches + batch_idx, lane_id);
  }

  constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
  auto outputs                 = cuda::std::array<T, out_per_thread>{};
  if constexpr (Mode == WarpReduceBatchedMode::SingleOut)
  {
    outputs[0] = warp_reduce_batched_t{temp_storage[logical_warp_id]}.Sum(inputs);
  }
  if constexpr (Mode == WarpReduceBatchedMode::ToBlocked)
  {
    warp_reduce_batched_t{temp_storage[logical_warp_id]}.SumToBlocked(inputs, outputs);
  }
  if constexpr (Mode == WarpReduceBatchedMode::ToStriped)
  {
    warp_reduce_batched_t{temp_storage[logical_warp_id]}.SumToStriped(inputs, outputs);
  }

  for (int idx = 0; idx < out_per_thread; ++idx)
  {
    const auto batch_idx =
      (Mode == WarpReduceBatchedMode::ToBlocked)
        ? (idx + lane_id * out_per_thread)
        : (idx * LogicalWarpThreads + lane_id);
    if (batch_idx < Batches)
    {
      output[logical_warp_id * Batches + batch_idx] = outputs[idx];
    }
  }
}

template <WarpReduceBatchedMode Mode,
          bool SyncPhysicalWarp,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp>
__global__ void
warp_reduce_batched_cond_part_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output, ReductionOp reduction_op)
{
  static_assert(LogicalWarpThreads < warp_size, "Need at least 2 logical warps per physical warp for this test");
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage[block_size / LogicalWarpThreads];

  const int tid               = threadIdx.x;
  const int logical_warp_id   = tid / LogicalWarpThreads;
  const int lane_id           = tid % LogicalWarpThreads;
  const bool is_participating = (logical_warp_id % 2 == 0);

  if (is_participating)
  {
    const int participant_idx = logical_warp_id / 2;

    auto inputs = cuda::std::array<T, Batches>{};
    for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
    {
      inputs[batch_idx] = input_md(participant_idx * Batches + batch_idx, lane_id);
    }

    constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
    auto outputs                 = cuda::std::array<T, out_per_thread>{};
    if constexpr (Mode == WarpReduceBatchedMode::SingleOut)
    {
      outputs[0] = warp_reduce_batched_t{temp_storage[logical_warp_id]}.Reduce(inputs, reduction_op);
    }
    if constexpr (Mode == WarpReduceBatchedMode::ToBlocked)
    {
      warp_reduce_batched_t{temp_storage[logical_warp_id]}.ReduceToBlocked(inputs, outputs, reduction_op);
    }
    if constexpr (Mode == WarpReduceBatchedMode::ToStriped)
    {
      warp_reduce_batched_t{temp_storage[logical_warp_id]}.ReduceToStriped(inputs, outputs, reduction_op);
    }

    for (int idx = 0; idx < out_per_thread; ++idx)
    {
      const auto batch_idx =
        (Mode == WarpReduceBatchedMode::ToBlocked)
          ? (idx + lane_id * out_per_thread)
          : (idx * LogicalWarpThreads + lane_id);
      if (batch_idx < Batches)
      {
        output[participant_idx * Batches + batch_idx] = outputs[idx];
      }
    }
  }
  // When SyncPhysicalWarp is true, the non-participating threads need to exit early to avoid expected deadlocks.
  if constexpr (!SyncPhysicalWarp)
  {
    // Keep non-participating threads from exiting early to check for unexpected deadlocks.
    __syncwarp();
  }
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // numeric(33): C: '=': conversion from 'int' to '_Ty', possible loss of data

template <typename T, typename ReductionOp>
void compute_host_reference(input_2d_mdspan_t<const T> input_md, cuda::std::span<T> output, ReductionOp op)
{
  const auto identity  = identity_v<ReductionOp, T>;
  const int batches    = input_md.extent(0);
  const int batch_size = input_md.extent(1);

  for (int batch_idx = 0; batch_idx < batches; ++batch_idx)
  {
    const auto iter   = cuda::make_transform_iterator(cuda::make_counting_iterator(0), [input_md, batch_idx](int idx) {
      return input_md(batch_idx, idx);
    });
    output[batch_idx] = static_cast<T>(std::accumulate(iter, iter + batch_size, identity, op));
  }
}

_CCCL_DIAG_POP

template <typename T, int N>
void gen_bounded_input(c2h::seed_t seed, c2h::device_vector<T>& d_input)
{
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    // Small positive range to minimize floating point error in reductions
    c2h::gen(seed, d_input, T(0.5), T(1.5));
  }
  else if constexpr (cuda::std::is_integral_v<T> && cuda::std::is_signed_v<T>)
  {
    // Avoid signed overflow when summing N elements
    const T gen_max = cuda::std::numeric_limits<T>::max() / static_cast<T>(N);
    c2h::gen(seed, d_input, -gen_max, gen_max);
  }
  else
  {
    c2h::gen(seed, d_input);
  }
}

template <WarpReduceBatchedMode Mode,
          int Batches,
          int LogicalWarpThreads,
          typename T,
          typename ReductionOp,
          bool ConvenienceOverload = false,
          bool CondParticipation   = false,
          bool SyncPhysicalWarp    = false>
void test_warp_reduce_batched(ReductionOp reduction_op = ReductionOp{})
{
  CAPTURE(c2h::type_name<T>(), Batches, LogicalWarpThreads);

  constexpr int num_logical_warps = block_size / LogicalWarpThreads;
  constexpr int total_batches     = CondParticipation ? num_logical_warps * Batches / 2 : num_logical_warps * Batches;
  constexpr int total_elements    = total_batches * LogicalWarpThreads;

  c2h::device_vector<T> d_input(total_elements);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(total_batches);

  input_2d_mdspan_t<T> d_input_md(thrust::raw_pointer_cast(d_input.data()), total_batches, LogicalWarpThreads);
  cuda::std::span<T> d_output_span(thrust::raw_pointer_cast(d_output.data()), total_batches);
  if constexpr (CondParticipation)
  {
    warp_reduce_batched_cond_part_kernel<Mode, SyncPhysicalWarp, Batches, LogicalWarpThreads>
      <<<1, block_size>>>(d_input_md, d_output_span, reduction_op);
  }
  else
  {
    if constexpr (ConvenienceOverload && cuda::std::is_same_v<ReductionOp, cuda::std::plus<>>)
    {
      sum_batched_kernel<Mode, SyncPhysicalWarp, Batches, LogicalWarpThreads>
        <<<1, block_size>>>(d_input_md, d_output_span);
    }
    else
    {
      warp_reduce_batched_kernel<Mode, SyncPhysicalWarp, Batches, LogicalWarpThreads>
        <<<1, block_size>>>(d_input_md, d_output_span, reduction_op);
    }
  }

  cudaError_t err = cudaPeekAtLastError();
  REQUIRE(err == cudaSuccess);

  err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);

  // Host-side: construct mdspans once; pass to reference and verify
  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;

  c2h::host_vector<T> h_reference(total_batches);
  input_2d_mdspan_t<const T> h_input_md(h_input.data(), total_batches, LogicalWarpThreads);
  cuda::std::span<T> h_reference_span(h_reference.data(), total_batches);

  compute_host_reference(h_input_md, h_reference_span, reduction_op);

  verify_results(h_reference, h_output);
}

#if TEST_TYPES == 0
using builtin_type_list = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t>;
#elif TEST_TYPES == 1
using builtin_type_list = c2h::type_list<cuda::std::int32_t, cuda::std::int64_t>;
#elif TEST_TYPES == 2
using builtin_type_list = c2h::type_list<float, double>;
#endif

#if TEST_TYPES == 0
using full_type_list = c2h::type_list<cuda::std::uint8_t, cuda::std::uint16_t, uchar3, short2>;
#elif TEST_TYPES == 1
using full_type_list =
  c2h::type_list<cuda::std::int32_t,
                 cuda::std::int64_t,
#  if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a
#  else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
                 >;
#elif TEST_TYPES == 2
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

using full_type_list = c2h::type_list<float, double, custom_t>;
#endif // TEST_TYPES

// (N, M) = (Batches, LogicalWarpThreads) for test parameterization
template <int X, int Y>
struct int_pair
{
  static constexpr int x = X;
  static constexpr int y = Y;
};

// N=M configurations (best performance)
using equal_nm_configs =
  c2h::type_list<int_pair<1, 1>, int_pair<2, 2>, int_pair<4, 4>, int_pair<8, 8>, int_pair<16, 16>, int_pair<32, 32>>;

// N!=M configurations
using unequal_nm_configs = c2h::type_list<
  int_pair<0, 32>,
  int_pair<1, 32>,
  int_pair<3, 32>,
  int_pair<3, 16>,
  int_pair<5, 16>,
  int_pair<4, 8>,
  int_pair<5, 8>,
  int_pair<6, 8>,
  int_pair<6, 4>,
  int_pair<7, 4>,
  int_pair<8, 4>,
  int_pair<9, 4>,
  int_pair<10, 4>,
  int_pair<1, 2>,
  int_pair<7, 2>,
  int_pair<0, 1>,
  int_pair<2, 1>>;

// Sub-warp configurations (LogicalWarpThreads < 32, at least 2 logical warps per physical warp)
using sub_warp_equal_configs = c2h::type_list<int_pair<2, 2>, int_pair<4, 4>, int_pair<8, 8>, int_pair<16, 16>>;

using sub_warp_unequal_configs = c2h::type_list<int_pair<3, 16>, int_pair<4, 8>, int_pair<6, 4>, int_pair<1, 2>>;

using unequal_nm_single_out_configs = c2h::type_list<int_pair<1, 32>, int_pair<3, 16>, int_pair<4, 8>, int_pair<1, 2>>;

C2H_TEST("WarpReduceBatched::Reduce N=M sum", "[warp][reduce][batched]", full_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::SingleOut, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST(
  "WarpReduceBatched::Reduce N!=M sum", "[warp][reduce][batched]", builtin_type_list, unequal_nm_single_out_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::SingleOut, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST(
  "WarpReduceBatched::Reduce max with over-syncing", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::maximum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::SingleOut,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::Reduce min with over-syncing",
         "[warp][reduce][batched]",
         builtin_type_list,
         unequal_nm_single_out_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::minimum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::SingleOut,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::Sum", "[warp][reduce][batched][convenience]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::SingleOut,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped N=M sum", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped N!=M sum", "[warp][reduce][batched]", builtin_type_list, unequal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped max with over-syncing",
         "[warp][reduce][batched]",
         builtin_type_list,
         equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::maximum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped min with over-syncing",
         "[warp][reduce][batched]",
         builtin_type_list,
         unequal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::minimum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::SumToStriped", "[warp][reduce][batched][convenience]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped with conditional participation N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped with conditional participation N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped with conditional participation and over-syncing N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::ReduceToStriped with conditional participation and over-syncing N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToStriped,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked N=M sum", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked N!=M sum", "[warp][reduce][batched]", builtin_type_list, unequal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked, num_batches, logical_warp_num_threads, value_t, op_t>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked max with over-syncing",
         "[warp][reduce][batched]",
         builtin_type_list,
         equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::maximum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked min with over-syncing",
         "[warp][reduce][batched]",
         builtin_type_list,
         unequal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::minimum<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = false;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::SumToBlocked", "[warp][reduce][batched][convenience]", builtin_type_list, equal_nm_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked with conditional participation N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked with conditional participation N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked with conditional participation and over-syncing N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}

C2H_TEST("WarpReduceBatched::ReduceToBlocked with conditional participation and over-syncing N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using value_t                          = c2h::get<0, TestType>;
  using op_t                             = cuda::std::plus<>;
  constexpr int num_batches              = c2h::get<1, TestType>::x;
  constexpr int logical_warp_num_threads = c2h::get<1, TestType>::y;
  constexpr bool convenience_overload    = false;
  constexpr bool cond_participation      = true;
  constexpr bool sync_physical_warp      = true;
  test_warp_reduce_batched<WarpReduceBatchedMode::ToBlocked,
                           num_batches,
                           logical_warp_num_threads,
                           value_t,
                           op_t,
                           convenience_overload,
                           cond_participation,
                           sync_physical_warp>();
}
