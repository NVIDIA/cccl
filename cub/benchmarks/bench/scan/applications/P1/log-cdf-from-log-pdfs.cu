// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/cmath>

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

namespace impl
{
/*
 * Given a sequence of logarithms of probability mass function values,
 * compute sequence of logarithms of cumulative distribution function values.
 *
 *    log(CDF(n)) = log(\sum( PDF(k), 0 <=k <=n ))
 *
 * This is inclusive scan using logaddexp binary operator:
 *    logaddexp( logpdf1, logpdf2 ) := log( exp(logpdf1) + exp(logpdf2) )
 *         == max(logpdf1, logpdf2) + log( 1 + exp(-abs(logpdf1 - logpdf2)))
 *
 * The last reformulation allows avoid numerical accuracy issues
 * caused by underflows.
 *
 */

struct log_add_plus
{
  /* Operator is commutative and associative */
  template <typename T>
  T __host__ __device__ operator()(T v1, T v2)
  {
    T max12 = cuda::maximum{}(v1, v2);
    T min12 = cuda::minimum{}(v1, v2);
    T exp   = cuda::std::exp(min12 - max12);
    return max12 + cuda::std::log1p(exp);
  }
};

template <typename T>
struct log_pdf_builder
{
  T mu;
  T norm;
  cuda::std::size_t n;

  T __host__ __device__ operator()(cuda::std::size_t i) const
  {
    return -mu * static_cast<T>(n - i) + norm;
  }
};

template <typename T>
[[nodiscard]] bool validate(const thrust::device_vector<T>& output, cudaStream_t stream)
{
  cudaStreamSynchronize(stream);

  thrust::host_vector<T> h_output(output);
  auto elements = h_output.size();
  // test is designed so that last element of prefix scan sequence should be close to log(1.0) == 0.0
  bool check = cuda::std::abs(h_output[elements - 1])
             < cuda::std::sqrt(static_cast<T>(1 + elements)) * cuda::std::numeric_limits<T>::epsilon();

  return check;
}
}; // namespace impl

template <typename FloatingPointT, typename OffsetT>
static void inclusive_scan(nvbench::state& state, nvbench::type_list<FloatingPointT, OffsetT>)
{
  static_assert(cuda::std::is_floating_point_v<FloatingPointT>);

  using wrapped_init_t = cub::NullType;
  using value_t        = FloatingPointT;
  using input_t        = const value_t*;
  using output_t       = value_t*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
  using op_t           = impl::log_add_plus;
  using accum_t        = value_t;

#if !TUNE_BASE
  using policy_t = policy_hub_t<accum_t>;
  using dispatch_t =
    cub::DispatchScan<input_t, output_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_t, output_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  auto mu             = static_cast<value_t>(state.get_float64("Mu{io}"));

  auto norm = cuda::std::log1p(-cuda::std::exp(-mu)) - cuda::std::log1p(-cuda::std::exp(-mu * elements));

  thrust::device_vector<value_t> input(elements, thrust::no_init);

  cudaStream_t bench_stream = state.get_cuda_stream();

  auto naturals_it = cuda::counting_iterator(cuda::std::size_t{0});
  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(naturals_it),
    input.begin(),
    elements,
    impl::log_pdf_builder<value_t>{mu, norm, elements},
    bench_stream);

  thrust::device_vector<value_t> output(elements, thrust::no_init);

  input_t d_input   = thrust::raw_pointer_cast(input.data());
  output_t d_output = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<value_t>(elements, "Size");
  state.add_global_memory_writes<value_t>(elements);

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), bench_stream);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_tmp, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), launch.get_stream());
  });

  // for validation, use
  // assert(impl::validate(output, bench_stream));
}

#ifdef TUNE_T
using fp_types = nvbench::type_list<TUNE_T>;
#else
using fp_types = nvbench::type_list<float, double>;
#endif

NVBENCH_BENCH_TYPES(inclusive_scan, NVBENCH_TYPE_AXES(fp_types, offset_types))
  .set_name("app-logcdf-from-logpdf")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_float64_axis("Mu{io}", {1e-4f});
