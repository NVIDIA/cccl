//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>

#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/data_manipulation.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

constexpr uint8_t fill_byte    = 1;
constexpr uint32_t buffer_size = 42;

int get_expected_value(uint8_t pattern_byte)
{
  int result;
  memset(&result, pattern_byte, sizeof(int));
  return result;
}

template <typename Result>
void check_result_and_erase(cudax::stream_ref stream, Result&& result, uint8_t pattern_byte = fill_byte)
{
  int expected = get_expected_value(pattern_byte);

  stream.wait();
  for (int& i : result)
  {
    CUDAX_REQUIRE(i == expected);
    i = 0;
  }
}

namespace cuda::experimental
{

// Need a type that goes through all launch_transform steps, but its not a contigious_range
struct weird_buffer
{
  const cuda::mr::pinned_memory_resource& resource;
  int* data;
  std::size_t size;

  weird_buffer(const cuda::mr::pinned_memory_resource& res, std::size_t s)
      : resource(res)
      , data((int*) res.allocate(s * sizeof(int)))
      , size(s)
  {}

  ~weird_buffer()
  {
    resource.deallocate(data, size);
  }

  weird_buffer(const weird_buffer&) = delete;
  weird_buffer(weird_buffer&&)      = delete;

  struct transform_result
  {
    int* data;
    std::size_t size;

    using __as_kernel_arg = cuda::std::span<int>;

    operator cuda::std::span<int>()
    {
      return {data, size};
    }
  };

  _CCCL_NODISCARD_FRIEND transform_result __cudax_launch_transform(cuda::stream_ref, const weird_buffer& self) noexcept
  {
    return {self.data, self.size};
  }
};

static_assert(std::is_same_v<cudax::as_kernel_arg_t<cudax::weird_buffer>, cuda::std::span<int>>);

} // namespace cuda::experimental

TEST_CASE("Fill", "[data_manipulation]")
{
  cudax::stream stream;
  SECTION("Host resource")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(host_resource, buffer_size);

    cudax::fill_bytes(stream, buffer, fill_byte);

    check_result_and_erase(stream, cuda::std::span(buffer));
  }

  SECTION("Device resource")
  {
    cuda::mr::device_memory_resource device_resource;
    cudax::uninitialized_buffer<int, cuda::mr::device_accessible> buffer(device_resource, buffer_size);
    cudax::fill_bytes(stream, buffer, fill_byte);

    std::vector<int> host_vector(42);
    CUDART(
      cudaMemcpyAsync(host_vector.data(), buffer.data(), buffer.size() * sizeof(int), cudaMemcpyDefault, stream.get()));

    check_result_and_erase(stream, host_vector);
  }
  SECTION("Launch transform")
  {
    cuda::mr::pinned_memory_resource host_resource;
    cudax::weird_buffer buffer(host_resource, buffer_size);

    cudax::fill_bytes(stream, buffer, fill_byte);
    check_result_and_erase(stream, cuda::std::span(buffer.data, buffer.size));
  }
}

TEST_CASE("Copy", "[data_manipulation]")
{
  cudax::stream stream;

  SECTION("Device resource")
  {
    cudax::mr::async_memory_resource device_resource;
    std::vector<int> host_vector(buffer_size);

    {
      cudax::uninitialized_async_buffer<int, cuda::mr::device_accessible> buffer(device_resource, stream, buffer_size);
      cudax::fill_bytes(stream, buffer, fill_byte);

      cudax::copy_bytes(stream, buffer, host_vector);
      check_result_and_erase(stream, host_vector);

      cudax::copy_bytes(stream, std::move(buffer), host_vector);
      check_result_and_erase(stream, host_vector);
    }
    {
      cudax::uninitialized_async_buffer<int, cuda::mr::device_accessible> not_yet_const_buffer(
        device_resource, stream, buffer_size);
      cudax::fill_bytes(stream, not_yet_const_buffer, fill_byte);

      const auto& const_buffer = not_yet_const_buffer;

      cudax::copy_bytes(stream, const_buffer, host_vector);
      check_result_and_erase(stream, host_vector);

      cudax::copy_bytes(stream, const_buffer, std::move(cuda::std::span(host_vector)));
      check_result_and_erase(stream, host_vector);
    }
  }

  SECTION("Host and managed resource")
  {
    cuda::mr::managed_memory_resource managed_resource;
    cuda::mr::pinned_memory_resource host_resource;

    {
      cudax::uninitialized_buffer<int, cuda::mr::host_accessible> host_buffer(host_resource, buffer_size);
      cudax::uninitialized_buffer<int, cuda::mr::device_accessible> device_buffer(managed_resource, buffer_size);

      cudax::fill_bytes(stream, host_buffer, fill_byte);

      cudax::copy_bytes(stream, host_buffer, device_buffer);
      check_result_and_erase(stream, device_buffer);

      cudax::copy_bytes(stream, std::move(cuda::std::span(host_buffer)), device_buffer);
      check_result_and_erase(stream, device_buffer);
    }

    {
      cudax::uninitialized_buffer<int, cuda::mr::host_accessible> not_yet_const_host_buffer(host_resource, buffer_size);
      cudax::uninitialized_buffer<int, cuda::mr::device_accessible> device_buffer(managed_resource, buffer_size);
      cudax::fill_bytes(stream, not_yet_const_host_buffer, fill_byte);

      const auto& const_host_buffer = not_yet_const_host_buffer;

      cudax::copy_bytes(stream, const_host_buffer, device_buffer);
      check_result_and_erase(stream, device_buffer);

      cudax::copy_bytes(stream, std::move(cuda::std::span(const_host_buffer)), device_buffer);
      check_result_and_erase(stream, device_buffer);
    }
  }
  SECTION("Launch transform")
  {
    cudax::stream stream;

    cuda::mr::pinned_memory_resource host_resource;
    cudax::weird_buffer input(host_resource, buffer_size);
    cudax::weird_buffer output(host_resource, buffer_size);

    memset(input.data, fill_byte, input.size * sizeof(int));

    cudax::copy_bytes(stream, input, output);
    check_result_and_erase(stream, cuda::std::span(output.data, output.size));
  }

  SECTION("Asymetric size")
  {
    cudax::stream stream;

    cuda::mr::pinned_memory_resource host_resource;
    cudax::uninitialized_buffer<int, cuda::mr::host_accessible> host_buffer(host_resource, 1);
    cudax::fill_bytes(stream, host_buffer, fill_byte);

    ::std::vector<int> vec(buffer_size, 0xdeadbeef);

    cudax::copy_bytes(stream, host_buffer, vec);
    stream.wait();

    CUDAX_REQUIRE(vec[0] == get_expected_value(fill_byte));
    CUDAX_REQUIRE(vec[1] == 0xdeadbeef);
  }
}
