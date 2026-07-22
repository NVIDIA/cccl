// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_segmented_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#if _CCCL_HAS_NVFP16()
#  include <cuda_fp16.h>
#endif

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_TRANSPOSE trp 0:1:1
// %RANGE% TUNE_LOAD ld 0:1:1

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using ScanByKeyPolicyT = cub::AgentScanByKeyPolicy<
      TUNE_THREADS,
      TUNE_ITEMS,
      // TODO Tune
      TUNE_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      cub::BLOCK_SCAN_WARP_SCANS,
      TUNE_STORE_ALGORITHM>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

namespace impl
{
template <typename T>
thrust::device_vector<T> generate_data(std::size_t n)
{
#if _CCCL_HAS_NVFP16()
  if constexpr (cuda::std::is_same_v<T, __half>)
  {
    // Generate as float, then convert to __half
    thrust::device_vector<float> temp = generate(n);
    thrust::device_vector<__half> result(n);
    thrust::transform(temp.begin(), temp.end(), result.begin(), [] __device__(float f) {
      return __float2half(f);
    });
    return result;
  }
  else
#endif
  {
    return generate(n);
  }
}

template <typename T>
struct s5_op_segmented
{
  __host__ __device__ cuda::std::tuple<T, T> operator()(cuda::std::tuple<T, T> x, cuda::std::tuple<T, T> y) const
  {
    const auto& [x_A, x_Bu] = x;
    const auto& [y_A, y_Bu] = y;

    return {y_A * x_A, y_A * x_Bu + y_Bu};
  }
};

// Helper struct to hold the lambda functors outside the function
struct column_major_transform
{
  int nrows;
  int ncols;

  __host__ __device__ int operator()(int k) const
  {
    int row = k % nrows;
    int col = k / nrows;
    return row * ncols + col;
  }
};
}; // namespace impl

template <typename T, typename OffsetT, typename StateDim>
static void segmented_scan(nvbench::state& state, nvbench::type_list<T, OffsetT, StateDim>)
{
  using wrapped_init_t      = cub::NullType;
  constexpr int state_dim   = StateDim::value;
  using value_t             = T;
  using op_t                = impl::s5_op_segmented<value_t>;
  using transformed_input_t = cuda::transform_iterator<impl::column_major_transform, cuda::counting_iterator<int>>;
  using permuted_input_t    = cuda::permutation_iterator<value_t*, transformed_input_t>;
  using permuted_output_t   = cuda::permutation_iterator<value_t*, transformed_input_t>;

  using input_it_t  = cuda::zip_iterator<permuted_input_t, permuted_input_t>;
  using output_it_t = cuda::zip_iterator<permuted_output_t, permuted_output_t>;

  using begin_offset_it_t = OffsetT*;
  using end_offset_it_t   = OffsetT*;

  using accum_t  = cuda::std::tuple<value_t, value_t>;
  using offset_t = cub::detail::common_iterator_value_t<begin_offset_it_t, end_offset_it_t>;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    input_it_t,
    output_it_t,
    begin_offset_it_t,
    end_offset_it_t,
    begin_offset_it_t,
    op_t,
    wrapped_init_t,
    accum_t,
    cub::ForceInclusive::Yes,
    offset_t,
    policy_t>;
#else
  using dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    input_it_t,
    output_it_t,
    begin_offset_it_t,
    end_offset_it_t,
    begin_offset_it_t,
    op_t,
    wrapped_init_t,
    accum_t,
    cub::ForceInclusive::Yes,
    offset_t>;
#endif

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto total_size     = elements * state_dim;
  cudaStream_t bench_stream = state.get_cuda_stream().get_stream();

  thrust::device_vector<value_t> input_A  = impl::generate_data<value_t>(total_size);
  thrust::device_vector<value_t> input_Bu = impl::generate_data<value_t>(total_size);
  thrust::device_vector<value_t> output_A(total_size, thrust::no_init);
  thrust::device_vector<value_t> output_Bu(total_size, thrust::no_init);

  const int nrows = elements;
  const int ncols = state_dim;

  auto col_major_iter =
    cuda::make_transform_iterator(cuda::make_counting_iterator(0), impl::column_major_transform{nrows, ncols});

  auto A_in_iter  = cuda::make_permutation_iterator(thrust::raw_pointer_cast(input_A.data()), col_major_iter);
  auto Bu_in_iter = cuda::make_permutation_iterator(thrust::raw_pointer_cast(input_Bu.data()), col_major_iter);
  auto input_iter = cuda::make_zip_iterator(A_in_iter, Bu_in_iter);

  auto A_output_iter  = cuda::make_permutation_iterator(thrust::raw_pointer_cast(output_A.data()), col_major_iter);
  auto Bu_output_iter = cuda::make_permutation_iterator(thrust::raw_pointer_cast(output_Bu.data()), col_major_iter);
  auto output_iter    = cuda::make_zip_iterator(A_output_iter, Bu_output_iter);

  thrust::device_vector<offset_t> begin_offsets(state_dim, thrust::no_init);
  thrust::device_vector<offset_t> end_offsets(state_dim, thrust::no_init);

  thrust::sequence(begin_offsets.begin(), begin_offsets.end(), 0, nrows);
  thrust::sequence(end_offsets.begin(), end_offsets.end(), nrows, nrows);

  state.add_element_count(elements);
  state.add_global_memory_reads<value_t>(total_size, "A_in");
  state.add_global_memory_reads<value_t>(total_size, "Bu_in");
  state.add_global_memory_writes<value_t>(total_size, "A_out");
  state.add_global_memory_writes<value_t>(total_size, "Bu_out");

  size_t tmp_size;
  dispatch_t::dispatch(
    nullptr,
    tmp_size,
    input_iter,
    output_iter,
    state_dim,
    thrust::raw_pointer_cast(begin_offsets.data()),
    thrust::raw_pointer_cast(end_offsets.data()),
    thrust::raw_pointer_cast(begin_offsets.data()),
    op_t{},
    wrapped_init_t{},
    bench_stream);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    dispatch_t::dispatch(
      d_tmp,
      tmp_size,
      input_iter,
      output_iter,
      state_dim,
      thrust::raw_pointer_cast(begin_offsets.data()),
      thrust::raw_pointer_cast(end_offsets.data()),
      thrust::raw_pointer_cast(begin_offsets.data()),
      op_t{},
      wrapped_init_t{},
      launch.get_stream());
  });
}

#ifdef TUNE_T
using fp_types = nvbench::type_list<TUNE_T>;
#else
#  if _CCCL_HAS_NVFP16()
using fp_types = nvbench::type_list<__half, float, double>;
#  else
using fp_types = nvbench::type_list<float, double>;
#  endif
#endif

using state_dim_types = nvbench::type_list<std::integral_constant<int, 40>>;

NVBENCH_BENCH_TYPES(segmented_scan, NVBENCH_TYPE_AXES(fp_types, offset_types, state_dim_types))
  .set_name("s5-segmented-scan")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "StateDim{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 24, 2));
