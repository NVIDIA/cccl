//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __DATA_MANIPULATION_COMMON__
#define __DATA_MANIPULATION_COMMON__

#include <cuda/memory_resource>

#include <cuda/experimental/buffer.cuh>
#include <cuda/experimental/data_manipulation.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <catch2/catch.hpp>
#include <utility.cuh>

inline constexpr uint8_t fill_byte    = 1;
inline constexpr uint32_t buffer_size = 42;

inline int get_expected_value(uint8_t pattern_byte)
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

// Need a type that goes through all launch_transform steps, but is not a contiguous_range
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

#endif // __DATA_MANIPULATION_COMMON__