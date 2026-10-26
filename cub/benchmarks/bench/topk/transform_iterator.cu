// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_topk.cuh>
#include <cub/util_type.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

// DeviceTopK is the device algorithm whose shipped tuning selects BLOCK_LOAD_VECTORIZE. The input is a
// transform_iterator over const char*, so the vectorized load copies the underlying bytes and the functor runs when
// the loaded items are read back.
struct widen
{
  [[nodiscard]] _CCCL_HOST_DEVICE uint32_t operator()(char value) const
  {
    return static_cast<uint32_t>(static_cast<unsigned char>(value));
  }
};

// Same tile shape as the shipped uint32 policy, with the load forced to BLOCK_LOAD_DIRECT.
struct direct_load_policy
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(::cuda::compute_capability cc) const
  {
    auto policy = cub::detail::topk::policy_selector_from_types<uint32_t, cub::NullType, int64_t, int>{}(cc);
    return cub::detail::topk::topk_policy{
      policy.threads_per_block,
      policy.items_per_thread,
      cub::BLOCK_LOAD_DIRECT,
      policy.scan_algorithm,
      policy.bits_per_pass};
  }
};

template <class Env>
void max_keys(nvbench::state& state, int64_t elements, int selected_elements, Env env)
{
  thrust::device_vector<char> input(static_cast<size_t>(elements), thrust::no_init);
  thrust::device_vector<uint32_t> output(static_cast<size_t>(selected_elements), thrust::no_init);

  const auto in = cuda::transform_iterator<widen, const char*>{thrust::raw_pointer_cast(input.data()), widen{}};
  auto* out     = thrust::raw_pointer_cast(output.data());

  state.add_element_count(elements, "NumElements");
  state.add_element_count(selected_elements, "NumSelectedElements");
  // Traffic is the underlying char storage, not the uint32 key the functor returns.
  state.add_global_memory_reads<char>(elements, "InputKeys");
  state.add_global_memory_writes<uint32_t>(selected_elements, "OutputKeys");

  size_t temp_size{};
  cub::DeviceTopK::MaxKeys(nullptr, temp_size, in, out, elements, selected_elements, env);
  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);

  auto* temp_storage = thrust::raw_pointer_cast(temp.data());
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env_with_stream = cuda::std::execution::env{cuda::stream_ref{launch.get_stream().get_stream()}, env};
    cub::DeviceTopK::MaxKeys(temp_storage, temp_size, in, out, elements, selected_elements, env_with_stream);
  });
}

void topk_transform(nvbench::state& state)
{
  const auto elements    = state.get_int64("Elements{io}");
  constexpr int selected = 1024;
  const bool vectorize   = state.get_string("Load") == "vectorize";

  if (selected >= elements)
  {
    state.skip("Selected count must be smaller than the input.");
    return;
  }

  auto requirements =
    cuda::execution::require(cuda::execution::determinism::not_guaranteed, cuda::execution::output_ordering::unsorted);
  if (vectorize)
  {
    max_keys(state, elements, selected, cuda::std::execution::env{requirements});
  }
  else
  {
    max_keys(
      state, elements, selected, cuda::std::execution::env{requirements, cuda::execution::tune(direct_load_policy{})});
  }
}

NVBENCH_BENCH(topk_transform)
  .set_name("transform")
  .add_string_axis("Load", {"vectorize", "direct"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(20, 28, 4));
