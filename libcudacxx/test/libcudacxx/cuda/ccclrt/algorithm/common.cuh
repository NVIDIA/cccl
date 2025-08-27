//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH
#define TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH

#include <cuda/algorithm>
#include <cuda/memory_resource>
#include <cuda/std/mdspan>
#include <cuda/stream>

#include <testing.cuh>
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
void check_result_and_erase(cuda::stream_ref stream, Result&& result, uint8_t pattern_byte = fill_byte)
{
  int expected = get_expected_value(pattern_byte);

  stream.sync();
  for (int& i : result)
  {
    CCCLRT_REQUIRE(i == expected);
    i = 0;
  }
}

enum class test_buffer_type
{
  pinned,
  device,
  managed
};

// Temporary test type until we move the buffer types to libcu++
template <typename T>
struct test_buffer
{
  test_buffer_type type;
  T* data_ptr;
  std::size_t buffer_size;

  test_buffer(test_buffer_type type, std::size_t size)
      : type(type)
      , data_ptr(nullptr)
      , buffer_size(size)
  {
    cuda::__ensure_current_context ctx_setter{cuda::device_ref{0}};
    if (type == test_buffer_type::pinned)
    {
      _CCCL_TRY_CUDA_API(cudaMallocHost, "Failed to allocate pinned memory", &data_ptr, size * sizeof(T));
    }
    else if (type == test_buffer_type::device)
    {
      _CCCL_TRY_CUDA_API(cudaMalloc, "Failed to allocate device memory", &data_ptr, size * sizeof(T));
    }
    else if (type == test_buffer_type::managed)
    {
      _CCCL_TRY_CUDA_API(cudaMallocManaged, "Failed to allocate managed memory", &data_ptr, size * sizeof(T));
    }
  }

  ~test_buffer()
  {
    if (data_ptr)
    {
      cuda::__ensure_current_context ctx_setter{cuda::device_ref{0}};
      if (type == test_buffer_type::pinned)
      {
        _CCCL_TRY_CUDA_API(cudaFreeHost, "Failed to free pinned memory", data_ptr);
      }
      else if (type == test_buffer_type::device)
      {
        _CCCL_TRY_CUDA_API(cudaFree, "Failed to free device memory", data_ptr);
      }
      else if (type == test_buffer_type::managed)
      {
        _CCCL_TRY_CUDA_API(cudaFree, "Failed to free managed memory", data_ptr);
      }
    }
  }

  test_buffer(const test_buffer&) = delete;
  test_buffer(test_buffer&& other)
      : type(other.type)
      , data_ptr(other.data_ptr)
      , buffer_size(other.buffer_size)
  {
    other.data_ptr    = nullptr;
    other.buffer_size = 0;
  }

  test_buffer& operator=(const test_buffer&) = delete;
  test_buffer& operator=(test_buffer&& other)
  {
    ::cuda::std::exchange(type, other.type);
    ::cuda::std::exchange(data_ptr, other.data_ptr);
    ::cuda::std::exchange(buffer_size, other.buffer_size);
    return *this;
  }

  T* begin() const
  {
    return data_ptr;
  }

  T* end() const
  {
    return data_ptr + buffer_size;
  }

  T* data() const
  {
    return data_ptr;
  }

  std::size_t size() const
  {
    return buffer_size;
  }

  std::size_t size_bytes() const
  {
    return buffer_size * sizeof(T);
  }

  operator cuda::std::span<const T>() const
  {
    return {data_ptr, buffer_size};
  }

  operator cuda::std::span<T>()
  {
    return {data_ptr, buffer_size};
  }
};

template <typename Layout = cuda::std::layout_right, typename Extents>
auto make_buffer_for_mdspan(Extents extents, char value = 0)
{
  auto mapping = typename Layout::template mapping<decltype(extents)>{extents};

  test_buffer<int> buffer(test_buffer_type::pinned, mapping.required_span_size());

  memset(buffer.data(), value, buffer.size_bytes());

  return buffer;
}

inline auto create_fake_strided_mdspan()
{
  cuda::std::dextents<size_t, 3> dynamic_extents{1, 2, 3};
  cuda::std::array<size_t, 3> strides{12, 4, 1};
#if _CCCL_CUDACC_BELOW(12, 6)
  auto map = cuda::std::layout_stride::mapping{dynamic_extents, strides};
#else
  cuda::std::layout_stride::mapping map{dynamic_extents, strides};
#endif
  return cuda::std::mdspan<int, decltype(dynamic_extents), cuda::std::layout_stride>(nullptr, map);
};

#endif // TEST_LIBCUDACXX_CCCLRT_ALGORITHM_COMMON_CUH
