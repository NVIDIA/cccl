// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/device/device_for.cuh>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:16:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 64:1024:64

#if !TUNE_BASE
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::ForPolicy
  {
    return {TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD};
  }
};
#endif // !TUNE_BASE

template <class T>
struct op_t
{
  const T* in;
  T* out;

  __device__ void operator()(int i) const
  {
    out[i] = in[i] + T{1};
  }
};

template <class T, class OffsetT>
void bulk(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  const auto elements = static_cast<OffsetT>(state.get_int64("Elements{io}"));

  thrust::device_vector<T> in(elements, T{42});
  thrust::device_vector<T> out(elements, thrust::no_init);

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_writes<T>(elements);

  op_t<T> op{thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data())};

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
    _CCCL_TRY_RUNTIME_API(cub::DeviceFor::Bulk, "Bulk failed", static_cast<int>(elements), op, env);
  });
}

NVBENCH_BENCH_TYPES(bulk, NVBENCH_TYPE_AXES(fundamental_types, nvbench::type_list<int32_t>))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
