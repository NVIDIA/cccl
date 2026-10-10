// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define CCCL_IGNORE_DEPRECATED_API

// The unsupported-architecture check is deferred to dispatch so this benchmark compiles for every configured
// architecture. Must precede the CUB include.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/device/device_batched_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>

#include <nvbench_helper.cuh>

#include "iterators.cuh"

// Batched top-k loads each segment's keys with a warp-transpose BlockLoad. Top-k selects numeric keys, so this sweep
// keeps the scalar synthesizing iterators and leaves zip inputs out. Each segment is a window of a counting sequence.
// The window length stays under the algorithm's per-segment limit.
inline constexpr int synthesizing_batched_topk_segment_size = 4096;
inline constexpr int synthesizing_batched_topk_k            = 32;

template <class Iterator>
struct synthesizing_batched_topk_segment_t
{
  Iterator base;

  _CCCL_HOST_DEVICE Iterator operator()(::cuda::std::int64_t segment) const
  {
    using difference_t = decltype(base - base);
    return base + static_cast<difference_t>(segment * synthesizing_batched_topk_segment_size);
  }
};

template <class ValueType>
struct synthesizing_batched_topk_output_t
{
  ValueType* base;

  _CCCL_HOST_DEVICE ValueType* operator()(::cuda::std::int64_t segment) const
  {
    return base + segment * synthesizing_batched_topk_k;
  }
};

// The cluster agent value-initializes its segment layout, which contains the input iterator. That agent is selected for
// 128-bit keys. thrust::shuffle_iterator has no default constructor, so those two types are covered by the other
// benchmarks and left out of this one.
using synthesizing_thrust_shuffle_topk_value_types =
  nvbench::type_list<cuda::std::int32_t, cuda::std::int64_t, cuda::std::uint32_t, cuda::std::uint64_t>;

using synthesizing_scalar_input_types = typename synthesizing_concat_inputs<
  synthesizing_inputs_for_t<cuda_counting_t>,
  synthesizing_inputs_for_t<thrust_counting_t>,
  synthesizing_inputs_for_t<cuda_constant_t>,
  synthesizing_inputs_for_t<thrust_constant_t>,
  synthesizing_inputs_for_t<cuda_shuffle_t, synthesizing_shuffle_value_types>,
  synthesizing_inputs_for_t<thrust_shuffle_t, synthesizing_thrust_shuffle_topk_value_types>,
  synthesizing_inputs_for_t<cuda_strided_counting_t>,
  synthesizing_inputs_for_t<thrust_strided_counting_t>,
  synthesizing_inputs_for_t<cuda_transform_counting_t>,
  synthesizing_inputs_for_t<thrust_transform_counting_t>,
  synthesizing_inputs_for_t<cuda_zip_transform_counting_t>>::type;

template <class Input>
void batched_topk(nvbench::state& state, nvbench::type_list<Input>)
try
{
  const auto elements        = static_cast<::cuda::std::int64_t>(state.get_int64("Elements{io}"));
  const auto num_segments    = elements / synthesizing_batched_topk_segment_size;
  auto input                 = Input::make(elements);
  using value_t              = cub::detail::it_value_t<decltype(input)>;
  const auto output_elements = static_cast<std::size_t>(num_segments * synthesizing_batched_topk_k);
  auto output                = synthesizing_make_output<value_t>(output_elements);
  const auto keys_in         = cuda::transform_iterator{
    cuda::counting_iterator<::cuda::std::int64_t>{0}, synthesizing_batched_topk_segment_t<decltype(input)>{input}};
  const auto keys_out = cuda::transform_iterator{
    cuda::counting_iterator<::cuda::std::int64_t>{0},
    synthesizing_batched_topk_output_t<value_t>{thrust::raw_pointer_cast(output.data())}};

  state.add_element_count(elements);
  state.add_global_memory_writes<value_t>(output_elements);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch,
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::tie_break::unspecified,
                               cuda::execution::output_ordering::unsorted));
    _CCCL_TRY_RUNTIME_API(
      cub::DeviceBatchedTopK::MaxKeys,
      "MaxKeys failed",
      keys_in,
      keys_out,
      cuda::args::constant<synthesizing_batched_topk_segment_size>{},
      cuda::args::constant<synthesizing_batched_topk_k>{},
      cuda::args::immediate{num_segments},
      env);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(batched_topk, NVBENCH_TYPE_AXES(synthesizing_scalar_input_types))
  .set_name("base")
  .set_type_axes_names({"Input{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
