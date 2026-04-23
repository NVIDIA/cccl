// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_adjacent_difference.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32

#if !TUNE_BASE
struct policy_selector_t
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id) const
    -> cub::detail::adjacent_difference::adjacent_difference_policy
  {
    return {TUNE_THREADS_PER_BLOCK,
            TUNE_ITEMS_PER_THREAD,
            cub::BLOCK_LOAD_WARP_TRANSPOSE,
            cub::LOAD_CA,
            cub::BLOCK_STORE_WARP_TRANSPOSE};
  }
};
#endif // !TUNE_BASE

template <class T, class OffsetT>
void left(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using input_it_t      = const T*;
  using output_it_t     = T*;
  using difference_op_t = ::cuda::std::minus<>;
  using offset_t        = cub::detail::choose_offset_t<OffsetT>;

  const auto elements         = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(elements, thrust::no_init);

  input_it_t d_in   = thrust::raw_pointer_cast(in.data());
  output_it_t d_out = thrust::raw_pointer_cast(out.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector_t{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceAdjacentDifference::SubtractLeftCopy,
      "SubtractLeftCopy failed",
      d_in,
      d_out,
      static_cast<offset_t>(elements),
      difference_op_t{},
      env);
  });
}

using types = nvbench::type_list<int32_t>;

NVBENCH_BENCH_TYPES(left, NVBENCH_TYPE_AXES(types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
