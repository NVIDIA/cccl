// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_reduce_batched.cuh>

#include <cuda/cmath>
#include <cuda/functional>
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

/***********************************************************************************************************************
 * Constants
 **********************************************************************************************************************/

inline constexpr int warp_size = 32;

// 2D layout: num_batches x batch_size (row-major). Single type with dynamic extents for kernels and host.
template <typename T>
using input_2d_mdspan_t = cuda::std::mdspan<T, cuda::std::dextents<int, 2>>;

/***********************************************************************************************************************
 * Batched Reduction Kernel
 **********************************************************************************************************************/

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
__global__ void
warp_reduce_batched_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output, ReductionOp reduction_op)
{
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage;
  const int tid             = threadIdx.x;
  const int logical_warp_id = tid / LogicalWarpThreads;
  const int lane_id         = tid % LogicalWarpThreads;

  T inputs[Batches];
  for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
  {
    inputs[batch_idx] = input_md(logical_warp_id * Batches + batch_idx, lane_id);
  }

  constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
  T outputs[out_per_thread];
  warp_reduce_batched_t{temp_storage}.Reduce(inputs, outputs, reduction_op);

  for (int idx = 0; idx < out_per_thread; ++idx)
  {
    const auto batch_idx = idx * LogicalWarpThreads + lane_id;
    if (batch_idx < Batches)
    {
      output[logical_warp_id * Batches + batch_idx] = outputs[idx];
    }
  }
}

template <typename T, int Batches, int LogicalWarpThreads>
__global__ void sum_batched_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output)
{
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage;
  const int tid             = threadIdx.x;
  const int logical_warp_id = tid / LogicalWarpThreads;
  const int lane_id         = tid % LogicalWarpThreads;

  T inputs[Batches];
  for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
  {
    inputs[batch_idx] = input_md(logical_warp_id * Batches + batch_idx, lane_id);
  }

  constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
  T outputs[out_per_thread];
  warp_reduce_batched_t{temp_storage}.Sum(inputs, outputs);

  for (int idx = 0; idx < out_per_thread; ++idx)
  {
    const auto batch_idx = idx * LogicalWarpThreads + lane_id;
    if (batch_idx < Batches)
    {
      output[logical_warp_id * Batches + batch_idx] = outputs[idx];
    }
  }
}

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
__global__ void
warp_reduce_batched_lane_mask_kernel(input_2d_mdspan_t<T> input_md, cuda::std::span<T> output, ReductionOp reduction_op)
{
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage;
  const int tid             = threadIdx.x;
  const int logical_warp_id = tid / LogicalWarpThreads;
  const int lane_id         = tid % LogicalWarpThreads;

  // Each logical warp uses a mask containing only its own lanes
  const cuda::std::uint32_t lane_mask = cuda::bitmask(logical_warp_id * LogicalWarpThreads, LogicalWarpThreads);

  T inputs[Batches];
  for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
  {
    inputs[batch_idx] = input_md(logical_warp_id * Batches + batch_idx, lane_id);
  }

  constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
  T outputs[out_per_thread];
  warp_reduce_batched_t{temp_storage}.Reduce(inputs, outputs, reduction_op, lane_mask);

  for (int idx = 0; idx < out_per_thread; ++idx)
  {
    const auto batch_idx = idx * LogicalWarpThreads + lane_id;
    if (batch_idx < Batches)
    {
      output[logical_warp_id * Batches + batch_idx] = outputs[idx];
    }
  }
}

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
__global__ void warp_reduce_batched_half_warps_kernel(
  input_2d_mdspan_t<T> input_md, cuda::std::span<T> output, ReductionOp reduction_op)
{
  static_assert(LogicalWarpThreads < warp_size, "Need at least 2 logical warps for this test");
  using warp_reduce_batched_t = cub::WarpReduceBatched<T, Batches, LogicalWarpThreads>;
  __shared__ typename warp_reduce_batched_t::TempStorage temp_storage;
  constexpr int num_logical_warps = warp_size / LogicalWarpThreads;

  const int tid               = threadIdx.x;
  const int logical_warp_id   = tid / LogicalWarpThreads;
  const int lane_id           = tid % LogicalWarpThreads;
  const bool is_participating = (logical_warp_id % 2 == 0);

  // Build a single mask shared by all participating threads: union of all even-indexed logical warp lanes
  cuda::std::uint32_t lane_mask = 0;
#pragma unroll
  for (int w = 0; w < num_logical_warps; w += 2)
  {
    lane_mask |= cuda::bitmask(w * LogicalWarpThreads, LogicalWarpThreads);
  }

  if (is_participating)
  {
    const int participant_idx = logical_warp_id / 2;

    T inputs[Batches];
    for (int batch_idx = 0; batch_idx < Batches; ++batch_idx)
    {
      inputs[batch_idx] = input_md(participant_idx * Batches + batch_idx, lane_id);
    }

    constexpr int out_per_thread = cuda::ceil_div(Batches, LogicalWarpThreads);
    T outputs[out_per_thread];
    warp_reduce_batched_t{temp_storage}.Reduce(inputs, outputs, reduction_op, lane_mask);

    for (int idx = 0; idx < out_per_thread; ++idx)
    {
      const auto batch_idx = idx * LogicalWarpThreads + lane_id;
      if (batch_idx < Batches)
      {
        output[participant_idx * Batches + batch_idx] = outputs[idx];
      }
    }
  }
  __syncwarp();
}

/***********************************************************************************************************************
 * Host Reference
 **********************************************************************************************************************/

template <typename T, typename ReductionOp>
void compute_host_reference(input_2d_mdspan_t<const T> input_md, cuda::std::span<T> output, ReductionOp op)
{
  const int batches    = input_md.extent(0);
  const int batch_size = input_md.extent(1);

  for (int batch_idx = 0; batch_idx < batches; ++batch_idx)
  {
    T result = input_md(batch_idx, 0);
    for (int idx = 1; idx < batch_size; ++idx)
    {
      result = op(result, input_md(batch_idx, idx));
    }
    output[batch_idx] = result;
  }
}

/***********************************************************************************************************************
 * Test Helpers
 **********************************************************************************************************************/

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

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
void test_warp_reduce_batched(ReductionOp reduction_op)
{
  CAPTURE(c2h::type_name<T>(), Batches, LogicalWarpThreads);

  constexpr int num_logical_warps = warp_size / LogicalWarpThreads;
  constexpr int total_batches     = num_logical_warps * Batches;
  constexpr int total_elements    = total_batches * LogicalWarpThreads;

  c2h::device_vector<T> d_input(total_elements);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(total_batches);

  input_2d_mdspan_t<T> d_input_md(thrust::raw_pointer_cast(d_input.data()), total_batches, LogicalWarpThreads);
  cuda::std::span<T> d_output_span(thrust::raw_pointer_cast(d_output.data()), total_batches);

  warp_reduce_batched_kernel<Batches, LogicalWarpThreads, T><<<1, warp_size>>>(d_input_md, d_output_span, reduction_op);

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

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
void test_warp_reduce_batched_lane_mask(ReductionOp reduction_op)
{
  CAPTURE(c2h::type_name<T>(), Batches, LogicalWarpThreads);

  constexpr int num_logical_warps = warp_size / LogicalWarpThreads;
  constexpr int total_batches     = num_logical_warps * Batches;
  constexpr int total_elements    = total_batches * LogicalWarpThreads;

  c2h::device_vector<T> d_input(total_elements);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(total_batches);

  input_2d_mdspan_t<T> d_input_md(thrust::raw_pointer_cast(d_input.data()), total_batches, LogicalWarpThreads);
  cuda::std::span<T> d_output_span(thrust::raw_pointer_cast(d_output.data()), total_batches);

  warp_reduce_batched_lane_mask_kernel<Batches, LogicalWarpThreads, T>
    <<<1, warp_size>>>(d_input_md, d_output_span, reduction_op);

  cudaError_t err = cudaPeekAtLastError();
  REQUIRE(err == cudaSuccess);

  err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);

  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;

  c2h::host_vector<T> h_reference(total_batches);
  input_2d_mdspan_t<const T> h_input_md(h_input.data(), total_batches, LogicalWarpThreads);
  cuda::std::span<T> h_reference_span(h_reference.data(), total_batches);

  compute_host_reference(h_input_md, h_reference_span, reduction_op);

  verify_results(h_reference, h_output);
}

template <int Batches, int LogicalWarpThreads, typename T, typename ReductionOp>
void test_warp_reduce_batched_half_warps(ReductionOp reduction_op)
{
  constexpr int num_logical_warps = warp_size / LogicalWarpThreads;
  static_assert(num_logical_warps >= 2, "Need at least 2 logical warps for half-warps test");
  constexpr int num_participating_warps = num_logical_warps / 2;

  CAPTURE(c2h::type_name<T>(), Batches, LogicalWarpThreads, num_participating_warps);

  constexpr int total_batches  = num_participating_warps * Batches;
  constexpr int total_elements = total_batches * LogicalWarpThreads;

  c2h::device_vector<T> d_input(total_elements);
  gen_bounded_input<T, LogicalWarpThreads>(C2H_SEED(10), d_input);

  c2h::device_vector<T> d_output(total_batches);

  input_2d_mdspan_t<T> d_input_md(thrust::raw_pointer_cast(d_input.data()), total_batches, LogicalWarpThreads);
  cuda::std::span<T> d_output_span(thrust::raw_pointer_cast(d_output.data()), total_batches);

  warp_reduce_batched_half_warps_kernel<Batches, LogicalWarpThreads, T>
    <<<1, warp_size>>>(d_input_md, d_output_span, reduction_op);

  cudaError_t err = cudaPeekAtLastError();
  REQUIRE(err == cudaSuccess);

  err = cudaDeviceSynchronize();
  REQUIRE(err == cudaSuccess);

  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;

  c2h::host_vector<T> h_reference(total_batches);
  input_2d_mdspan_t<const T> h_input_md(h_input.data(), total_batches, LogicalWarpThreads);
  cuda::std::span<T> h_reference_span(h_reference.data(), total_batches);

  compute_host_reference(h_input_md, h_reference_span, reduction_op);

  verify_results(h_reference, h_output);
}

/***********************************************************************************************************************
 * Type Lists
 **********************************************************************************************************************/

using builtin_type_list = c2h::type_list<std::uint8_t, std::uint16_t, std::int32_t, std::int64_t, float, double>;

using custom_type_list =
  c2h::type_list<c2h::custom_type_t<c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>,
                 c2h::custom_type_t<c2h::equal_comparable_t>,
                 uchar3,
#if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a
#else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4
#endif // _CCCL_CTK_AT_LEAST(13, 0)
                 >;

using full_type_list =
  c2h::type_list<std::uint8_t,
                 std::uint16_t,
                 std::int32_t,
                 std::int64_t,
                 float,
                 double,
                 c2h::custom_type_t<c2h::equal_comparable_t>,
                 uchar3>;

// (N, M) = (Batches, LogicalWarpThreads) for test parameterization
template <int X, int Y>
struct int_pair
{
  static constexpr int x = X;
  static constexpr int y = Y;
};

// N=M configurations (best performance)
using equal_nm_configs =
  // c2h::type_list<int_pair<32, 32>>;
  c2h::type_list<int_pair<2, 2>, int_pair<4, 4>, int_pair<8, 8>, int_pair<16, 16>, int_pair<32, 32>>;

// N!=M configurations
using unequal_nm_configs =
  c2h::type_list<int_pair<2, 32>,
                 int_pair<4, 16>,
                 int_pair<8, 16>,
                 int_pair<7, 16>,
                 int_pair<17, 32>,
                 int_pair<32, 16>,
                 int_pair<17, 16>,
                 int_pair<20, 16>,
                 int_pair<15, 8>>;

// Sub-warp configurations (LogicalWarpThreads < 32, at least 2 logical warps per physical warp)
using sub_warp_equal_configs = c2h::type_list<int_pair<2, 2>, int_pair<4, 4>, int_pair<8, 8>, int_pair<16, 16>>;

using sub_warp_unequal_configs =
  c2h::type_list<int_pair<4, 16>,
                 int_pair<4, 8>,
                 int_pair<7, 16>,
                 int_pair<8, 4>,
                 int_pair<17, 16>,
                 int_pair<20, 16>,
                 int_pair<15, 8>>;

/***********************************************************************************************************************
 * Test Cases
 **********************************************************************************************************************/

C2H_TEST("WarpReduceBatched::Reduce N=M sum", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<N, M, T>(cuda::std::plus<>{});
}

C2H_TEST("WarpReduceBatched::Reduce N!=M sum", "[warp][reduce][batched]", builtin_type_list, unequal_nm_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<N, M, T>(cuda::std::plus<>{});
}

C2H_TEST("WarpReduceBatched::Reduce max", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<N, M, T>(cuda::maximum<>{});
}

C2H_TEST("WarpReduceBatched::Reduce min", "[warp][reduce][batched]", builtin_type_list, equal_nm_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched<N, M, T>(cuda::minimum<>{});
}

C2H_TEST("WarpReduceBatched::Sum", "[warp][reduce][batched][convenience]", builtin_type_list, equal_nm_configs)
{
  using T                         = c2h::get<0, TestType>;
  constexpr int N                 = c2h::get<1, TestType>::x;
  constexpr int M                 = c2h::get<1, TestType>::y;
  constexpr int num_logical_warps = warp_size / M;
  constexpr int total_batches     = num_logical_warps * N;
  constexpr int total_elements    = total_batches * M;

  c2h::device_vector<T> d_input(total_elements);
  gen_bounded_input<T, M>(C2H_SEED(10), d_input);
  c2h::device_vector<T> d_output(total_batches);

  input_2d_mdspan_t<T> d_input_md(thrust::raw_pointer_cast(d_input.data()), total_batches, M);
  cuda::std::span<T> d_output_span(thrust::raw_pointer_cast(d_output.data()), total_batches);
  sum_batched_kernel<T, N, M><<<1, warp_size>>>(d_input_md, d_output_span);

  c2h::host_vector<T> h_input  = d_input;
  c2h::host_vector<T> h_output = d_output;
  c2h::host_vector<T> h_reference(total_batches);
  input_2d_mdspan_t<const T> h_input_md(h_input.data(), total_batches, M);
  cuda::std::span<T> h_reference_span(h_reference.data(), total_batches);
  compute_host_reference(h_input_md, h_reference_span, cuda::std::plus<>{});

  verify_results(h_reference, h_output);
}

C2H_TEST("WarpReduceBatched::Reduce lane_mask N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched_lane_mask<N, M, T>(cuda::std::plus<>{});
}

C2H_TEST("WarpReduceBatched::Reduce lane_mask N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched_lane_mask<N, M, T>(cuda::std::plus<>{});
}

C2H_TEST("WarpReduceBatched::Reduce half_warps N=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_equal_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched_half_warps<N, M, T>(cuda::std::plus<>{});
}

C2H_TEST("WarpReduceBatched::Reduce half_warps N!=M sum",
         "[warp][reduce][batched][lane_mask]",
         builtin_type_list,
         sub_warp_unequal_configs)
{
  using T         = c2h::get<0, TestType>;
  constexpr int N = c2h::get<1, TestType>::x;
  constexpr int M = c2h::get<1, TestType>::y;
  test_warp_reduce_batched_half_warps<N, M, T>(cuda::std::plus<>{});
}
