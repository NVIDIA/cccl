//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include <catch2/catch_test_macros.hpp>
#include <cccl/c/device_copy.h>

namespace
{
template <typename T>
class device_buffer
{
public:
  explicit device_buffer(std::size_t size)
  {
    void* raw_ptr = nullptr;
    CATCH_REQUIRE(cudaMalloc(&raw_ptr, size * sizeof(T)) == cudaSuccess);
    ptr_ = static_cast<T*>(raw_ptr);
  }

  device_buffer(const device_buffer&)            = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  ~device_buffer()
  {
    if (ptr_ != nullptr)
    {
      static_cast<void>(cudaFree(ptr_));
    }
  }

  T* get() const
  {
    return ptr_;
  }

private:
  T* ptr_ = nullptr;
};

struct device_copy_build_guard
{
  cccl_device_copy_build_result_t build{};

  ~device_copy_build_guard()
  {
    if (build.copy_fn != nullptr)
    {
      static_cast<void>(cccl_device_copy_cleanup(&build));
    }
  }
};
} // namespace

CATCH_TEST_CASE("DeviceCopy C API can execute rank-1 HostJIT copy", "[device_copy][hostjit]")
{
  int device = 0;
  CATCH_REQUIRE(cudaGetDevice(&device) == cudaSuccess);

  cudaDeviceProp properties{};
  CATCH_REQUIRE(cudaGetDeviceProperties(&properties, device) == cudaSuccess);

  constexpr std::size_t num_items = 1024;
  std::vector<std::int32_t> input(num_items);
  std::vector<std::int32_t> output(num_items, -1);
  for (std::size_t i = 0; i < num_items; ++i)
  {
    input[i] = static_cast<std::int32_t>(i * 3 + 7);
  }

  device_buffer<std::int32_t> d_input(num_items);
  device_buffer<std::int32_t> d_output(num_items);
  CATCH_REQUIRE(
    cudaMemcpy(d_input.get(), input.data(), num_items * sizeof(std::int32_t), cudaMemcpyHostToDevice) == cudaSuccess);

  const cccl_device_copy_axis_metadata_t shape[] = {{CCCL_DEVICE_COPY_AXIS_RUNTIME, 0}};
  const cccl_type_info value_type{sizeof(std::int32_t), alignof(std::int32_t), CCCL_INT32};
  const cccl_device_copy_build_spec_t spec{
    value_type, 1, shape, {CCCL_DEVICE_COPY_LAYOUT_RIGHT, nullptr}, {CCCL_DEVICE_COPY_LAYOUT_RIGHT, nullptr}};

  device_copy_build_guard device_copy;
  CATCH_REQUIRE(
    cccl_device_copy_build(
      &device_copy.build,
      spec,
      properties.major,
      properties.minor,
      TEST_CUB_PATH,
      TEST_THRUST_PATH,
      TEST_LIBCUDACXX_PATH,
      TEST_CTK_PATH)
    == CUDA_SUCCESS);

  const std::int64_t runtime_shape[] = {static_cast<std::int64_t>(num_items)};
  const cccl_device_copy_source_view_t source{d_input.get(), 0, runtime_shape, nullptr};
  const cccl_device_copy_destination_view_t destination{d_output.get(), 0, runtime_shape, nullptr};

  CATCH_REQUIRE(cccl_device_copy(device_copy.build, source, destination, nullptr) == CUDA_SUCCESS);
  CATCH_REQUIRE(
    cudaMemcpy(output.data(), d_output.get(), num_items * sizeof(std::int32_t), cudaMemcpyDeviceToHost) == cudaSuccess);
  CATCH_REQUIRE(output == input);

  CATCH_REQUIRE(cccl_device_copy_cleanup(&device_copy.build) == CUDA_SUCCESS);
}
