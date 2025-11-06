// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/tuple>

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
template <typename T>
using triplet_t = cuda::std::tuple<T, T, T>;

/* The triplet corresponds to strictly upper triangular elements of a unitriangular matrix
   A = [[1, a1, a12], [0, 1, a2], [0, 0, 1]], mapped to triplet [a1, a2, a12].

   The set of unitriangular matrix forms a group, with product induced by matrix multiplication,
   and the identity element corresponding to zero triplet.
*/
struct unitriangular_dim3_op
{
  // Scan operation: associative and non-commutative
  template <typename T>
  triplet_t<T> __host__ __device__ operator()(triplet_t<T> a, triplet_t<T> b) const
  {
    auto [a1, a2, a12] = a;
    auto [b1, b2, b12] = b;

    return {a1 + b1, a2 + b1, a12 + b12 + a1 * b2};
  }
};

// Utility operation to pack arguments into a triplet_t instance
struct pack_op
{
  template <typename T>
  triplet_t<T> __host__ __device__ operator()(T a1, T a2, T a12) const
  {
    return {a1, a2, a12};
  } // namespace impl
};

template <typename TupleT, typename ScanOpT>
bool validation(const thrust::device_vector<TupleT>& input,
                const thrust::device_vector<TupleT>& output,
                ScanOpT op,
                cudaStream_t stream)
{
  cudaStreamSynchronize(stream);

  using tuple_t = TupleT;
  thrust::host_vector<tuple_t> h_input(input);
  thrust::host_vector<tuple_t> h_output(output);

  auto elements = input.size();
  thrust::host_vector<tuple_t> h_reference(elements);

  h_reference[0] = h_input[0];
  for (std::size_t i = 1; i < elements; ++i)
  {
    h_reference[i] = op(h_reference[i - 1], h_input[i]);
  }

  return h_reference == h_output;
}
}; // namespace impl

template <typename T, typename OffsetT>
void benchmark_impl(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using wrapped_init_t = cub::NullType;
  using value_t        = T;
  using tuple_t        = impl::triplet_t<value_t>;
  using op_t           = impl::unitriangular_dim3_op;
  using accum_t        = tuple_t;
  using input_it_t     = const tuple_t*;
  using output_it_t    = tuple_t*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::
    DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No, policy_t>;
#else
  using dispatch_t =
    cub::DispatchScan<input_it_t, output_it_t, op_t, wrapped_init_t, offset_t, accum_t, cub::ForceInclusive::No>;
#endif

  const auto elements       = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  cudaStream_t bench_stream = state.get_cuda_stream().get_stream();

  thrust::device_vector<tuple_t> output(elements);
  thrust::device_vector<value_t> _input = generate(cuda::std::tuple_size_v<tuple_t> * elements);
  thrust::device_vector<tuple_t> input(elements);

  cub::DeviceTransform::Transform(
    cuda::std::make_tuple(cuda::strided_iterator(_input.begin(), std::size_t{3}),
                          cuda::strided_iterator(_input.begin() + 1, std::size_t{3}),
                          cuda::strided_iterator(_input.begin() + 2, std::size_t{3})),
    input.begin(),
    input.size(),
    impl::pack_op{},
    bench_stream);

  state.add_element_count(elements);
  state.add_global_memory_reads<tuple_t>(elements, "Size");
  state.add_global_memory_writes<tuple_t>(elements);

  auto d_input  = thrust::raw_pointer_cast(input.data());
  auto d_output = thrust::raw_pointer_cast(output.data());

  size_t tmp_size;
  dispatch_t::Dispatch(nullptr, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), bench_stream);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      d_tmp, tmp_size, d_input, d_output, op_t{}, wrapped_init_t{}, input.size(), launch.get_stream());
  });

  // for validation use (recommended for integral types and smallish input sizes)
  // assert(impl::validation(input, output, op_t{}, bench_stream));
}

#ifdef TUNE_T
using bench_types = nvbench::type_list<TUNE_T>;
#else
using bench_types = nvbench::type_list<nvbench::int32_t, nvbench::uint64_t, nvbench::float32_t, nvbench::float64_t>;
#endif

NVBENCH_BENCH_TYPES(benchmark_impl, NVBENCH_TYPE_AXES(bench_types, offset_types))
  .set_name("unitriangular-monoid")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
