// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/iterator>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include <look_back_helper.cuh>
#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS ipt 7:24:1
// %RANGE% TUNE_THREADS tpb 128:1024:32
// %RANGE% TUNE_MAGIC_NS ns 0:2048:4
// %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1
// %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5
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
      TUNE_STORE_ALGORITHM,
      delay_constructor_t>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

/* Given input sequence of values, compute sequence of
 * pairs corresponding to running minimum and running maximum values.
 */

/*! @brief Structure to hold minimum and maximum */
template <typename T>
struct MinMax
{
private:
  T m_min{cuda::std::numeric_limits<T>::max()};
  T m_max{cuda::std::numeric_limits<T>::min()};

public:
  __host__ __device__ MinMax() = default;
  __host__ __device__ MinMax(T minimum, T maximum)
      : m_min(minimum)
      , m_max(maximum)
  {}

  T __host__ __device__ minimum() const
  {
    return m_min;
  }
  T __host__ __device__ maximum() const
  {
    return m_max;
  }
};

/*! @brief Scan operator combining min-max pairs */
struct ScanOp
{
  template <typename T>
  MinMax<T> __host__ __device__ operator()(MinMax<T> v1, MinMax<T> v2) const
  {
    auto min_r = cuda::minimum{}(v1.minimum(), v2.minimum());
    auto max_r = cuda::maximum{}(v1.maximum(), v2.maximum());
    return {min_r, max_r};
  }
};

template <typename T>
struct EmbedOp
{
  MinMax<T> __host__ __device__ operator()(T v) const
  {
    return {v, v};
  }
};

template <typename T>
struct ExtractMin
{
  T __host__ __device__ operator()(MinMax<T> pair) const
  {
    return pair.minimum();
  }
};

template <typename T>
struct ExtractMax
{
  T __host__ __device__ operator()(MinMax<T> pair) const
  {
    return pair.maximum();
  }
};

template <typename T, typename OffsetT>
void benchmark_impl(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using wrapped_init_t = cub::NullType;
  using value_t        = T;
  using pair_t         = MinMax<value_t>;
  using op_t           = ScanOp;
  using accum_t        = pair_t;
  using input_raw_t    = const value_t*;
  using input_it_t     = cuda::transform_iterator<EmbedOp<value_t>, input_raw_t>;
  using output_it_t    = pair_t*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<pair_t> output(elements);
  thrust::device_vector<value_t> input = generate(elements);

  input_raw_t d_input  = thrust::raw_pointer_cast(input.data());
  output_it_t d_output = thrust::raw_pointer_cast(output.data());

  input_it_t inp_it(d_input, EmbedOp<value_t>{});

  state.add_element_count(elements);
  state.add_global_memory_reads<value_t>(elements, "Size");
  state.add_global_memory_writes<pair_t>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(
    nullptr, tmp_size, inp_it, d_output, op_t{}, wrapped_init_t{}, input.size(), state.get_cuda_stream().get_stream());

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(d_tmp, tmp_size, inp_it, d_output, op_t{}, wrapped_init_t{}, input.size(), launch.get_stream());
  });

  // verification

  thrust::device_vector<value_t> ref_mins(elements);
  thrust::device_vector<value_t> ref_maxs(elements);

  cub::DeviceScan::InclusiveScanInit(
    nullptr,
    tmp_size,
    d_input,
    ref_mins.begin(),
    cuda::minimum<>{},
    cuda::std::numeric_limits<value_t>::max(),
    input.size(),
    state.get_cuda_stream().get_stream());

  thrust::device_vector<nvbench::uint8_t> tmp1(tmp_size);
  nvbench::uint8_t* d_tmp1 = thrust::raw_pointer_cast(tmp1.data());

  cub::DeviceScan::InclusiveScanInit(
    d_tmp1,
    tmp_size,
    d_input,
    ref_mins.begin(),
    cuda::minimum<>{},
    cuda::std::numeric_limits<value_t>::max(),
    input.size(),
    state.get_cuda_stream().get_stream());

  cub::DeviceScan::InclusiveScanInit(
    nullptr,
    tmp_size,
    d_input,
    ref_maxs.begin(),
    cuda::minimum<>{},
    cuda::std::numeric_limits<value_t>::max(),
    input.size(),
    state.get_cuda_stream().get_stream());

  thrust::device_vector<nvbench::uint8_t> tmp2(tmp_size);
  nvbench::uint8_t* d_tmp2 = thrust::raw_pointer_cast(tmp2.data());

  cub::DeviceScan::InclusiveScanInit(
    d_tmp2,
    tmp_size,
    d_input,
    ref_maxs.begin(),
    cuda::maximum<>{},
    cuda::std::numeric_limits<value_t>::min(),
    input.size(),
    state.get_cuda_stream().get_stream());

  thrust::device_vector<value_t> computed_mins(elements);
  thrust::device_vector<value_t> computed_maxs(elements);

  cub::DeviceTransform::Transform(
    d_output, computed_mins.begin(), input.size(), ExtractMin<T>{}, state.get_cuda_stream().get_stream());
  cub::DeviceTransform::Transform(
    d_output, computed_maxs.begin(), input.size(), ExtractMax<T>{}, state.get_cuda_stream().get_stream());

  assert(computed_mins == ref_mins);
  assert(computed_maxs == ref_maxs);
}

using bench_types =
  nvbench::type_list<cuda::std::uint32_t,
                     nvbench::int32_t,
                     cuda::std::uint64_t,
                     nvbench::int64_t,
                     nvbench::float32_t,
                     nvbench::float64_t>;

NVBENCH_BENCH_TYPES(benchmark_impl, NVBENCH_TYPE_AXES(bench_types, offset_types))
  .set_name("running-min-max")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
