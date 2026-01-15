// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/device/dispatch/dispatch_reduce_nondeterministic.cuh>

#include <nvbench_helper.cuh>

#include <nvbench/range.cuh>
#include <nvbench/types.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 3:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_ITEMS_PER_VEC_LOAD_POW2 ipv 1:2:1

#if !TUNE_BASE
struct policy_selector
{
  _CCCL_API constexpr auto operator()(cuda::arch_id) const -> ::cub::reduce_policy
  {
    const auto [items, threads] = cub::detail::scale_mem_bound(TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD);
    const auto policy           = cub::agent_reduce_policy{
      threads,
      items,
      1 << TUNE_ITEMS_PER_VEC_LOAD_POW2,
      cub::BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
      cub::LOAD_DEFAULT};
    return {{}, {}, {}, policy}; // Only reduce_nondeterministic is used
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void nondeterministic_sum(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  using offset_t = cub::detail::choose_offset_t<OffsetT>;
  using op_t     = cuda::std::plus<>;
  using init_t   = T;

  // Retrieve axis parameters
  const auto elements = static_cast<std::size_t>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in = generate(elements);
  thrust::device_vector<T> out(1);

  auto d_in  = thrust::raw_pointer_cast(in.data());
  auto d_out = thrust::raw_pointer_cast(out.data());

  // Enable throughput calculations and add "Size" column to results.
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(1);

  auto transform_op = ::cuda::std::identity{};

  // Allocate temporary storage:
  std::size_t temp_size;
  cub::detail::reduce::dispatch_nondeterministic</* OverrideAccumT = */ T>(
    nullptr,
    temp_size,
    d_in,
    d_out,
    static_cast<offset_t>(elements),
    op_t{},
    init_t{},
    0 /* stream */,
    transform_op
#if !TUNE_BASE
    ,
    policy_selector{}
#endif
  );

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::reduce::dispatch_nondeterministic</* OverrideAccumT = */ T>(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      static_cast<offset_t>(elements),
      op_t{},
      init_t{},
      launch.get_stream(),
      transform_op
#if !TUNE_BASE
      ,
      policy_selector{}
#endif
    );
  });
}

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

NVBENCH_BENCH_TYPES(nondeterministic_sum, NVBENCH_TYPE_AXES(value_types, offset_types))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
