// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

namespace device_scan_concurrent_launch_test
{
struct interleaving_scan_launcher_factory;
}
#define CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY \
  ::device_scan_concurrent_launch_test::interleaving_scan_launcher_factory

#include <cub/device/device_scan.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cstddef>
#include <cstdint>

#include <c2h/catch2_test_helper.h>
#include <c2h/vector.h>

namespace device_scan_concurrent_launch_test
{
using value_t = std::int32_t;

struct interleaving_scan_state
{
  void* small_temp_storage{};
  std::size_t small_temp_storage_bytes{};
  value_t* small_input{};
  value_t* small_output{};
  int small_num_items{};
  cudaStream_t small_stream{};

  bool injected_small_scan{};
  bool running_small_scan{};
  cudaError_t small_scan_status{cudaErrorUnknown};
  std::size_t small_dynamic_smem{};
  std::size_t large_dynamic_smem{};
};

interleaving_scan_state* active_interleaving_scan_state{};

struct interleaving_scan_launcher : THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron
{
  using base_t = THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron;

  interleaving_scan_state* state{};

  CUB_RUNTIME_FUNCTION interleaving_scan_launcher(
    dim3 grid,
    dim3 block,
    std::size_t shared_mem,
    cudaStream_t stream,
    bool dependent_launch,
    interleaving_scan_state* state)
      : base_t(grid, block, shared_mem, stream, dependent_launch)
      , state(state)
  {}

  template <typename Kernel, typename... Args>
  CUB_RUNTIME_FUNCTION cudaError_t doit(Kernel kernel, Args const&... args) const
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   if (state != nullptr && !state->injected_small_scan)
                   {
                     state->injected_small_scan = true;
                     state->running_small_scan  = true;
                     state->small_scan_status   = cub::DeviceScan::ExclusiveSum(
                       state->small_temp_storage,
                       state->small_temp_storage_bytes,
                       state->small_input,
                       state->small_output,
                       state->small_num_items,
                       state->small_stream);
                     state->running_small_scan = false;

                     if (state->small_scan_status != cudaSuccess)
                     {
                       return state->small_scan_status;
                     }
                   }
                 }))

    return base_t::doit(kernel, args...);
  }
};

struct interleaving_scan_launcher_factory : cub::detail::TripleChevronFactory
{
  CUB_RUNTIME_FUNCTION interleaving_scan_launcher
  operator()(dim3 grid, dim3 block, std::size_t shared_mem, cudaStream_t stream, bool dependent_launch = false) const
  {
    interleaving_scan_state* state{};
    NV_IF_TARGET(NV_IS_HOST, ({
                   state = active_interleaving_scan_state;
                   if (state != nullptr && shared_mem != 0)
                   {
                     auto& recorded_smem =
                       state->running_small_scan ? state->small_dynamic_smem : state->large_dynamic_smem;
                     recorded_smem = shared_mem;
                   }
                 }))

    return {grid, block, shared_mem, stream, dependent_launch, state};
  }
};

struct interleaving_scan_scope
{
  explicit interleaving_scan_scope(interleaving_scan_state& state)
  {
    active_interleaving_scan_state = &state;
  }

  ~interleaving_scan_scope()
  {
    active_interleaving_scan_state = nullptr;
  }

  interleaving_scan_scope(interleaving_scan_scope const&)            = delete;
  interleaving_scan_scope& operator=(interleaving_scan_scope const&) = delete;
};

C2H_TEST("Device scan supports interleaved launches with different shared memory requirements", "[scan][device]")
{
  int device{};
  REQUIRE(cudaSuccess == cudaGetDevice(&device));

  int ptx_version{};
  REQUIRE(cudaSuccess == cub::PtxVersion(ptx_version, device));
  if (ptx_version < 1000)
  {
    return;
  }

  // These sizes come from a reduced reproducer for a TPC-H SF=30K failure on an NVL72 cluster. On
  // SM100, they select different Warpspeed stage counts while sharing the same scan-kernel instantiation.
  constexpr int large_num_items = 3'253'172;
  constexpr int small_num_items = 52;

  c2h::device_vector<value_t> large_input(large_num_items, value_t{});
  c2h::device_vector<value_t> large_output(large_num_items, value_t{-1});
  c2h::device_vector<value_t> small_input(small_num_items, value_t{});
  c2h::device_vector<value_t> small_output(small_num_items, value_t{-1});

  cudaStream_t large_stream{};
  cudaStream_t small_stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&large_stream));
  REQUIRE(cudaSuccess == cudaStreamCreate(&small_stream));

  std::size_t large_temp_storage_bytes{};
  std::size_t small_temp_storage_bytes{};
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      nullptr,
      large_temp_storage_bytes,
      large_input.data().get(),
      large_output.data().get(),
      large_num_items,
      large_stream));
  REQUIRE(
    cudaSuccess
    == cub::DeviceScan::ExclusiveSum(
      nullptr,
      small_temp_storage_bytes,
      small_input.data().get(),
      small_output.data().get(),
      small_num_items,
      small_stream));

  c2h::device_vector<std::uint8_t> large_temp_storage(large_temp_storage_bytes, thrust::no_init);
  c2h::device_vector<std::uint8_t> small_temp_storage(small_temp_storage_bytes, thrust::no_init);

  interleaving_scan_state state{
    small_temp_storage.data().get(),
    small_temp_storage_bytes,
    small_input.data().get(),
    small_output.data().get(),
    small_num_items,
    small_stream};

  cudaError_t large_scan_status{};
  {
    interleaving_scan_scope scope{state};
    large_scan_status = cub::DeviceScan::ExclusiveSum(
      large_temp_storage.data().get(),
      large_temp_storage_bytes,
      large_input.data().get(),
      large_output.data().get(),
      large_num_items,
      large_stream);
  }

  REQUIRE(state.injected_small_scan);
  REQUIRE(cudaSuccess == state.small_scan_status);
  INFO("Large scan failed with " << cudaGetErrorString(large_scan_status));
  REQUIRE(cudaSuccess == large_scan_status);
  REQUIRE(state.small_dynamic_smem > 0);
  REQUIRE(state.large_dynamic_smem > state.small_dynamic_smem);

  REQUIRE(cudaSuccess == cudaStreamSynchronize(small_stream));
  REQUIRE(cudaSuccess == cudaStreamSynchronize(large_stream));
  REQUIRE(value_t{} == small_output.back());
  REQUIRE(value_t{} == large_output.back());

  REQUIRE(cudaSuccess == cudaStreamDestroy(small_stream));
  REQUIRE(cudaSuccess == cudaStreamDestroy(large_stream));
}
} // namespace device_scan_concurrent_launch_test
