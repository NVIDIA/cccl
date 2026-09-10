//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include <catch2/catch_test_macros.hpp>
#include <cccl/c/device_copy.h>

namespace
{
template <class T>
class device_buffer
{
public:
  explicit device_buffer(std::size_t count)
  {
    void* raw_ptr = nullptr;
    CATCH_REQUIRE(cudaMalloc(&raw_ptr, count * sizeof(T)) == cudaSuccess);
    ptr_ = static_cast<T*>(raw_ptr);
  }

  ~device_buffer()
  {
    if (ptr_ != nullptr)
    {
      static_cast<void>(cudaFree(ptr_));
    }
  }

  device_buffer(const device_buffer&)            = delete;
  device_buffer& operator=(const device_buffer&) = delete;

  T* get() const
  {
    return ptr_;
  }

private:
  T* ptr_ = nullptr;
};

class device_copy_build_guard
{
public:
  device_copy_build_guard() = default;

  ~device_copy_build_guard()
  {
    if (build.copy_fn != nullptr)
    {
      static_cast<void>(cccl_device_copy_cleanup(&build));
    }
  }

  device_copy_build_guard(const device_copy_build_guard&)            = delete;
  device_copy_build_guard& operator=(const device_copy_build_guard&) = delete;
  device_copy_build_guard(device_copy_build_guard&&)                 = delete;
  device_copy_build_guard& operator=(device_copy_build_guard&&)      = delete;

  cccl_device_copy_build_result_t build{};
};

template <std::size_t Rank>
int64_t view_offset(const std::array<int64_t, Rank>& strides, const std::array<int64_t, Rank>& indices)
{
  int64_t offset = 0;
  for (std::size_t axis = 0; axis < Rank; ++axis)
  {
    offset += indices[axis] * strides[axis];
  }
  return offset;
}

template <std::size_t Rank>
std::array<cccl_device_copy_axis_metadata_t, Rank> runtime_axis_metadata()
{
  std::array<cccl_device_copy_axis_metadata_t, Rank> metadata{};
  metadata.fill({CCCL_DEVICE_COPY_AXIS_RUNTIME, 0});
  return metadata;
}

template <std::size_t Rank>
std::array<cccl_device_copy_axis_metadata_t, Rank> static_axis_metadata(const std::array<int64_t, Rank>& values)
{
  std::array<cccl_device_copy_axis_metadata_t, Rank> metadata{};
  for (std::size_t axis = 0; axis < Rank; ++axis)
  {
    metadata[axis] = {CCCL_DEVICE_COPY_AXIS_STATIC, values[axis]};
  }
  return metadata;
}
} // namespace

CATCH_TEST_CASE("C v2 DeviceCopy can copy runtime layout_stride views", "[device_copy][hostjit]")
{
  constexpr std::size_t rank = 2;
  constexpr std::array<int64_t, rank> shape{3, 4};
  constexpr std::array<int64_t, rank> source_strides{6, 1};
  constexpr std::array<int64_t, rank> destination_strides{1, 3};
  constexpr std::array<int64_t, rank> invalid_strides{0, 1};

  const auto shape_metadata  = runtime_axis_metadata<rank>();
  const auto stride_metadata = runtime_axis_metadata<rank>();

  std::vector<int> source_host(18, -1);
  std::vector<int> expected(12, 0);
  for (int64_t i = 0; i < shape[0]; ++i)
  {
    for (int64_t j = 0; j < shape[1]; ++j)
    {
      const int value = static_cast<int>(10 * i + j);
      source_host[static_cast<std::size_t>(i * source_strides[0] + j * source_strides[1])]        = value;
      expected[static_cast<std::size_t>(i * destination_strides[0] + j * destination_strides[1])] = value;
    }
  }

  device_buffer<int> d_source(source_host.size());
  device_buffer<int> d_destination(expected.size());
  std::vector<int> destination_host(expected.size(), 0);

  CATCH_REQUIRE(cudaMemcpy(d_source.get(), source_host.data(), source_host.size() * sizeof(int), cudaMemcpyHostToDevice)
                == cudaSuccess);
  CATCH_REQUIRE(cudaMemset(d_destination.get(), 0, expected.size() * sizeof(int)) == cudaSuccess);

  int device = 0;
  CATCH_REQUIRE(cudaGetDevice(&device) == cudaSuccess);
  cudaDeviceProp properties{};
  CATCH_REQUIRE(cudaGetDeviceProperties(&properties, device) == cudaSuccess);

  cccl_type_info value_type{};
  value_type.size      = sizeof(int);
  value_type.alignment = alignof(int);

  const cccl_device_copy_build_spec_t spec{
    value_type,
    rank,
    shape_metadata.data(),
    {CCCL_DEVICE_COPY_LAYOUT_STRIDE, stride_metadata.data()},
    {CCCL_DEVICE_COPY_LAYOUT_STRIDE, stride_metadata.data()}};

  device_copy_build_guard device_copy;
  CATCH_REQUIRE(
    cccl_device_copy_build_ex(
      &device_copy.build,
      spec,
      properties.major,
      properties.minor,
      TEST_CUB_PATH,
      TEST_THRUST_PATH,
      TEST_LIBCUDACXX_PATH,
      TEST_CTK_PATH,
      nullptr)
    == CUDA_SUCCESS);

  const cccl_device_copy_source_view_t source{d_source.get(), 0, shape.data(), source_strides.data()};
  const cccl_device_copy_destination_view_t destination{
    d_destination.get(), 0, shape.data(), destination_strides.data()};

  CATCH_REQUIRE(cccl_device_copy(device_copy.build, source, destination, nullptr) == CUDA_SUCCESS);
  CATCH_REQUIRE(
    cudaMemcpy(
      destination_host.data(), d_destination.get(), destination_host.size() * sizeof(int), cudaMemcpyDeviceToHost)
    == cudaSuccess);
  CATCH_REQUIRE(destination_host == expected);

  const cccl_device_copy_source_view_t invalid_source{d_source.get(), 0, shape.data(), invalid_strides.data()};
  CATCH_REQUIRE(cccl_device_copy(device_copy.build, invalid_source, destination, nullptr) == CUDA_ERROR_INVALID_VALUE);
}

CATCH_TEST_CASE("C v2 DeviceCopy can copy runtime layout_stride_relaxed views", "[device_copy][hostjit]")
{
  constexpr std::size_t rank = 3;

  constexpr std::array<int64_t, rank> shape{2, 3, 4};
  constexpr std::array<int64_t, rank> source_strides{-20, 5, 1};
  constexpr std::array<int64_t, rank> destination_strides{18, -5, 1};
  constexpr std::size_t source_first_element      = 20;
  constexpr std::size_t destination_first_element = 10;
  constexpr std::size_t source_storage_size       = 40;
  constexpr std::size_t destination_storage_size  = 40;

  std::vector<int> source_storage(source_storage_size);
  for (std::size_t i = 0; i < source_storage.size(); ++i)
  {
    source_storage[i] = static_cast<int>(1000 + i);
  }

  std::vector<int> destination_storage(destination_storage_size, -1);
  std::vector<int> expected_destination = destination_storage;

  for (int64_t i = 0; i < shape[0]; ++i)
  {
    for (int64_t j = 0; j < shape[1]; ++j)
    {
      for (int64_t k = 0; k < shape[2]; ++k)
      {
        const std::array<int64_t, rank> index{i, j, k};
        const auto source_index =
          static_cast<std::size_t>(static_cast<int64_t>(source_first_element) + view_offset(source_strides, index));
        const auto destination_index = static_cast<std::size_t>(
          static_cast<int64_t>(destination_first_element) + view_offset(destination_strides, index));
        expected_destination[destination_index] = source_storage[source_index];
      }
    }
  }

  device_buffer<int> d_source(source_storage.size());
  device_buffer<int> d_destination(destination_storage.size());

  CATCH_REQUIRE(
    cudaMemcpy(d_source.get(), source_storage.data(), source_storage.size() * sizeof(int), cudaMemcpyHostToDevice)
    == cudaSuccess);
  CATCH_REQUIRE(
    cudaMemcpy(
      d_destination.get(), destination_storage.data(), destination_storage.size() * sizeof(int), cudaMemcpyHostToDevice)
    == cudaSuccess);

  int current_device = 0;
  CATCH_REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);

  cudaDeviceProp props{};
  CATCH_REQUIRE(cudaGetDeviceProperties(&props, current_device) == cudaSuccess);

  auto shape_metadata              = runtime_axis_metadata<rank>();
  auto source_stride_metadata      = runtime_axis_metadata<rank>();
  auto destination_stride_metadata = runtime_axis_metadata<rank>();

  cccl_device_copy_build_spec_t spec{
    cccl_type_info{sizeof(int), alignof(int), CCCL_INT32},
    rank,
    shape_metadata.data(),
    cccl_device_copy_view_build_t{CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED, source_stride_metadata.data()},
    cccl_device_copy_view_build_t{CCCL_DEVICE_COPY_LAYOUT_STRIDE_RELAXED, destination_stride_metadata.data()}};

  device_copy_build_guard device_copy;
  CATCH_REQUIRE(
    cccl_device_copy_build(
      &device_copy.build,
      spec,
      props.major,
      props.minor,
      TEST_CUB_PATH,
      TEST_THRUST_PATH,
      TEST_LIBCUDACXX_PATH,
      TEST_CTK_PATH)
    == CUDA_SUCCESS);
  CATCH_REQUIRE(device_copy.build.shape != nullptr);
  CATCH_REQUIRE(device_copy.build.source_strides != nullptr);
  CATCH_REQUIRE(device_copy.build.destination_strides != nullptr);

  source_stride_metadata[0]      = {CCCL_DEVICE_COPY_AXIS_STATIC, 1};
  destination_stride_metadata[0] = {CCCL_DEVICE_COPY_AXIS_STATIC, 1};

  const cccl_device_copy_source_view_t source{
    d_source.get(), source_first_element * sizeof(int), shape.data(), source_strides.data()};
  const cccl_device_copy_destination_view_t destination{
    d_destination.get(), destination_first_element * sizeof(int), shape.data(), destination_strides.data()};

  CATCH_REQUIRE(cccl_device_copy(device_copy.build, source, destination, nullptr) == CUDA_SUCCESS);
  CATCH_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  CATCH_REQUIRE(
    cudaMemcpy(
      destination_storage.data(), d_destination.get(), destination_storage.size() * sizeof(int), cudaMemcpyDeviceToHost)
    == cudaSuccess);

  CATCH_REQUIRE(destination_storage == expected_destination);
}

CATCH_TEST_CASE("C v2 DeviceCopy can copy mixed static and runtime extents", "[device_copy][hostjit]")
{
  constexpr std::size_t rank = 3;

  constexpr std::array<int64_t, rank> shape{2, 3, 4};
  constexpr std::array<int64_t, rank> mismatched_static_shape{2, 3, 5};
  constexpr std::array<int64_t, rank> layout_right_strides{12, 4, 1};
  constexpr std::array<int64_t, rank> bad_layout_right_strides{1, 4, 12};
  constexpr std::size_t num_items = 24;

  std::vector<int> source_storage(num_items);
  for (std::size_t i = 0; i < source_storage.size(); ++i)
  {
    source_storage[i] = static_cast<int>(1000 + i);
  }
  std::vector<int> destination_storage(num_items, -1);

  device_buffer<int> d_source(source_storage.size());
  device_buffer<int> d_destination(destination_storage.size());

  CATCH_REQUIRE(
    cudaMemcpy(d_source.get(), source_storage.data(), source_storage.size() * sizeof(int), cudaMemcpyHostToDevice)
    == cudaSuccess);
  CATCH_REQUIRE(
    cudaMemcpy(
      d_destination.get(), destination_storage.data(), destination_storage.size() * sizeof(int), cudaMemcpyHostToDevice)
    == cudaSuccess);

  int current_device = 0;
  CATCH_REQUIRE(cudaGetDevice(&current_device) == cudaSuccess);

  cudaDeviceProp props{};
  CATCH_REQUIRE(cudaGetDeviceProperties(&props, current_device) == cudaSuccess);

  auto shape_metadata = static_axis_metadata(shape);
  shape_metadata[1]   = {CCCL_DEVICE_COPY_AXIS_RUNTIME, 0};

  cccl_device_copy_build_spec_t spec{
    cccl_type_info{sizeof(int), alignof(int), CCCL_INT32},
    rank,
    shape_metadata.data(),
    cccl_device_copy_view_build_t{CCCL_DEVICE_COPY_LAYOUT_RIGHT, nullptr},
    cccl_device_copy_view_build_t{CCCL_DEVICE_COPY_LAYOUT_RIGHT, nullptr}};

  device_copy_build_guard device_copy;
  CATCH_REQUIRE(
    cccl_device_copy_build(
      &device_copy.build,
      spec,
      props.major,
      props.minor,
      TEST_CUB_PATH,
      TEST_THRUST_PATH,
      TEST_LIBCUDACXX_PATH,
      TEST_CTK_PATH)
    == CUDA_SUCCESS);
  CATCH_REQUIRE(device_copy.build.shape != nullptr);
  CATCH_REQUIRE(device_copy.build.source_strides == nullptr);
  CATCH_REQUIRE(device_copy.build.destination_strides == nullptr);

  const cccl_device_copy_source_view_t source{d_source.get(), 0, shape.data(), layout_right_strides.data()};
  const cccl_device_copy_destination_view_t destination{
    d_destination.get(), 0, shape.data(), layout_right_strides.data()};

  CATCH_REQUIRE(cccl_device_copy(device_copy.build, source, destination, nullptr) == CUDA_SUCCESS);
  CATCH_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  CATCH_REQUIRE(
    cudaMemcpy(
      destination_storage.data(), d_destination.get(), destination_storage.size() * sizeof(int), cudaMemcpyDeviceToHost)
    == cudaSuccess);
  CATCH_REQUIRE(destination_storage == source_storage);

  const cccl_device_copy_source_view_t bad_source_strides{
    d_source.get(), 0, shape.data(), bad_layout_right_strides.data()};
  CATCH_REQUIRE(
    cccl_device_copy(device_copy.build, bad_source_strides, destination, nullptr) == CUDA_ERROR_INVALID_VALUE);

  const cccl_device_copy_destination_view_t bad_destination_strides{
    d_destination.get(), 0, shape.data(), bad_layout_right_strides.data()};
  CATCH_REQUIRE(
    cccl_device_copy(device_copy.build, source, bad_destination_strides, nullptr) == CUDA_ERROR_INVALID_VALUE);

  const cccl_device_copy_source_view_t mismatched_source{d_source.get(), 0, mismatched_static_shape.data(), nullptr};
  const cccl_device_copy_destination_view_t mismatched_destination{
    d_destination.get(), 0, mismatched_static_shape.data(), nullptr};
  CATCH_REQUIRE(cccl_device_copy(device_copy.build, mismatched_source, mismatched_destination, nullptr)
                == CUDA_ERROR_INVALID_VALUE);
}
