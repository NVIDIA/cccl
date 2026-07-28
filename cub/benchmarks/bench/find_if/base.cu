// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_find.cuh>

#include <thrust/count.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/find.h>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_LOAD ld 0:2:1
// %RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK_POW2 tpb 6:10:1

#if !TUNE_BASE
#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename T>
struct bench_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(::cuda::compute_capability) const
    -> cub::detail::find::find_policy
  {
    return cub::detail::find::find_policy{
      (1 << TUNE_THREADS_PER_BLOCK_POW2), cub::Nominal4BItemsToItems<T>(TUNE_ITEMS_PER_THREAD), 4, TUNE_LOAD_MODIFIER};
  }
};
#endif // !TUNE_BASE

template <typename T, typename OffsetT>
void find_if(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  T val = 1;
  // set up input
  const auto elements       = static_cast<OffsetT>(state.get_int64("Elements"));
  const auto common_prefix  = state.get_float64("MismatchAt");
  const auto mismatch_point = static_cast<OffsetT>(elements * common_prefix);

  thrust::device_vector<T> dinput(elements, thrust::no_init);
  thrust::fill(dinput.begin(), dinput.begin() + mismatch_point, 0);
  thrust::fill(dinput.begin() + mismatch_point, dinput.end(), val);
  thrust::device_vector<OffsetT> d_result(1, thrust::no_init);

  state.add_global_memory_reads<T>(mismatch_point);
  state.add_global_memory_writes<OffsetT>(1);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(bench_policy_selector<T>{})
#endif // !TUNE_BASE
    );
    _CCCL_TRY_CUDA_API(
      cub::DeviceFind::FindIf,
      "FindIf failed",
      thrust::raw_pointer_cast(dinput.data()),
      thrust::raw_pointer_cast(d_result.data()),
      cuda::equal_to_value<T>(val),
      static_cast<OffsetT>(dinput.size()),
      env);
  });
}

NVBENCH_BENCH_TYPES(find_if, NVBENCH_TYPE_AXES(fundamental_types, offset_types))
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_float64_axis("MismatchAt", std::vector{1.0, 0.5, 0.0});
