//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/detail/utility/atomic.cuh>

#include <cuda_runtime_api.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

struct atomic_test_values
{
  alignas(4)::cuda::std::uint8_t cas8;
  ::cuda::std::uint8_t guard8_0;
  ::cuda::std::uint8_t guard8_1;
  ::cuda::std::uint8_t guard8_2;

  alignas(4)::cuda::std::uint16_t cas16;
  ::cuda::std::uint16_t guard16;

  ::cuda::std::int32_t value32;
  ::cuda::std::int64_t value64;
  ::cuda::std::int32_t maximum32;
  ::cuda::std::int32_t cas32;
  ::cuda::std::int32_t stored32;
  ::cuda::std::int32_t loaded32;
};

__global__ void test_atomic_kernel(atomic_test_values* values)
{
  const auto rank = static_cast<::cuda::std::int32_t>(threadIdx.x);

  (void) cudax::cuco::detail::__atomic_fetch_add<::cuda::thread_scope_device>(&values->value32, ::cuda::std::int32_t{1});
  (void) cudax::cuco::detail::__atomic_fetch_add<::cuda::thread_scope_system>(&values->value64, ::cuda::std::int64_t{1});
  (void) cudax::cuco::detail::__atomic_fetch_max<::cuda::thread_scope_block>(&values->maximum32, rank);

  auto expected8 = ::cuda::std::uint8_t{0};
  (void) cudax::cuco::detail::__atomic_compare_exchange<::cuda::thread_scope_device>(
    &values->cas8, expected8, static_cast<::cuda::std::uint8_t>(rank + 1));

  auto expected16 = ::cuda::std::uint16_t{0};
  (void) cudax::cuco::detail::__atomic_compare_exchange<::cuda::thread_scope_device>(
    &values->cas16, expected16, static_cast<::cuda::std::uint16_t>(rank + 1));

  auto expected32 = ::cuda::std::int32_t{0};
  (void) cudax::cuco::detail::__atomic_compare_exchange<::cuda::thread_scope_device>(
    &values->cas32, expected32, rank + 1);

  __syncthreads();
  if (rank == 0)
  {
    cudax::cuco::detail::__atomic_store<::cuda::thread_scope_device>(&values->stored32, ::cuda::std::int32_t{9});
    values->loaded32 = cudax::cuco::detail::__atomic_load<::cuda::thread_scope_device>(&values->stored32);
  }
}

C2H_TEST("cuco plain CUDA atomics preserve values and packed neighbors", "[atomic]")
{
  atomic_test_values host_values{};
  host_values.guard8_0 = 0xA5;
  host_values.guard8_1 = 0x5A;
  host_values.guard8_2 = 0xC3;
  host_values.guard16  = 0xA55A;
  host_values.stored32 = 3;

  atomic_test_values* device_values{};
  REQUIRE_CUDART(::cudaMalloc(&device_values, sizeof(atomic_test_values)));
  REQUIRE_CUDART(::cudaMemcpy(device_values, &host_values, sizeof(atomic_test_values), ::cudaMemcpyHostToDevice));

  constexpr int block_size = 128;
  test_atomic_kernel<<<1, block_size>>>(device_values);
  REQUIRE_CUDART(::cudaGetLastError());
  REQUIRE_CUDART(::cudaDeviceSynchronize());
  REQUIRE_CUDART(::cudaMemcpy(&host_values, device_values, sizeof(atomic_test_values), ::cudaMemcpyDeviceToHost));
  REQUIRE_CUDART(::cudaFree(device_values));

  REQUIRE(host_values.cas8 >= 1);
  REQUIRE(host_values.cas8 <= block_size);
  REQUIRE(host_values.guard8_0 == 0xA5);
  REQUIRE(host_values.guard8_1 == 0x5A);
  REQUIRE(host_values.guard8_2 == 0xC3);
  REQUIRE(host_values.cas16 >= 1);
  REQUIRE(host_values.cas16 <= block_size);
  REQUIRE(host_values.guard16 == 0xA55A);
  REQUIRE(host_values.value32 == block_size);
  REQUIRE(host_values.value64 == block_size);
  REQUIRE(host_values.maximum32 == block_size - 1);
  REQUIRE(host_values.cas32 >= 1);
  REQUIRE(host_values.cas32 <= block_size);
  REQUIRE(host_values.stored32 == 9);
  REQUIRE(host_values.loaded32 == 9);
}
