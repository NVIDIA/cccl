// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

#include "thrust/iterator/transform_iterator.h"

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#ifndef TUNE_BASE
#  define TUNE_ITEMS_PER_VEC_LOAD (1 << TUNE_ITEMS_PER_VEC_LOAD_POW2)
#endif

#if !TUNE_BASE
struct policy_selector
{
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> ::cub::reduce_policy
  {
    const auto [items, threads] = cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD);
    const auto policy           = cub::agent_reduce_policy{
      threads, items, 1 << TUNE_ITEMS_PER_VEC_LOAD_POW2, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {policy, policy, policy, policy};
  }
};
#endif // !TUNE_BASE

template <class T>
struct square_t
{
  __host__ __device__ T operator()(const T& x) const
  {
    return x * x;
  }
};

#define USE_TRANSPOSE_ITERATOR 0

#if USE_TRANSPOSE_ITERATOR
template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using accum_t        = T;
  using input_it_t     = thrust::transform_iterator<square_t<T>, typename thrust::device_vector<T>::iterator>;
  using output_it_t    = T*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
  using output_t       = T;
  using init_t         = T;
  using reduction_op_t = ::cuda::std::plus<>;
  using transform_op_t = square_t<T>;

  using dispatch_t = cub::DispatchReduce<
    input_it_t,
    output_it_t,
    offset_t,
    reduction_op_t,
    init_t,
    accum_t
#  if !TUNE_BASE
    ,
    cuda::std::identity,
    policy_hub_t<accum_t, offset_t>
#  endif // TUNE_BASE
    >;

  // Retrieve axis parameters
  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  input_it_t d_in   = thrust::make_transform_iterator(in.begin(), square_t<T>{});
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // Allocate temporary storage:
  std::size_t temp_size;
  dispatch_t::Dispatch(
    nullptr, temp_size, d_in, d_out, static_cast<offset_t>(elements), reduction_op_t{}, init_t{}, 0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::Dispatch(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      static_cast<offset_t>(elements),
      reduction_op_t{},
      init_t{},
      launch.get_stream());
  });
}
#else
template <typename T, typename OffsetT>
void reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
  using init_t         = T;
  using reduction_op_t = ::cuda::std::plus<>;
  using transform_op_t = square_t<T>;

  // Retrieve axis parameters
  const auto elements         = static_cast<offset_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::detail::reduce::dispatch</* OverrideAccumT = */ T>(
    nullptr,
    temp_size,
    d_in,
    d_out,
    elements,
    reduction_op_t{},
    init_t{},
    0 /* stream */,
    transform_op_t{}
#  if !TUNE_BASE
    ,
    policy_selector{}
#  endif
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch</* OverrideAccumT = */ T>(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      elements,
      reduction_op_t{},
      init_t{},
      0 /* stream */,
      transform_op_t{}
#  if !TUNE_BASE
      ,
      policy_selector{}
#  endif
    );
  });
}
#endif

NVBENCH_BENCH_TYPES(reduce, NVBENCH_TYPE_AXES(all_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
