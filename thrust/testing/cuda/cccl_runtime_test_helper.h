#pragma once

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/initializer_list>
#include <cuda/std/span>
#include <cuda/stream>

#include <cuda_runtime_api.h>

#include <unittest/unittest.h>

#include <cstdio>
#include <vector>

namespace test
{
inline cuda::device_ref current_test_device()
{
  int device = 0;
  ASSERT_EQUAL(cudaSuccess, cudaGetDevice(&device));
  return cuda::device_ref{device};
}

inline auto single_thread_config()
{
  return cuda::make_config(cuda::make_hierarchy(cuda::grid_dims(1), cuda::block_dims<1>()));
}

__device__ inline void assert_device(bool condition, const char* expression, const char* file, int line)
{
  if (!condition)
  {
    printf("Device assertion failed: %s (%s:%d)\n", expression, file, line);
    __trap();
  }
}

template <typename Buffer>
void assert_equal(cuda::stream_ref stream, Buffer& buffer, cuda::std::initializer_list<int> expected)
{
  std::vector<int> actual(buffer.size());
  cuda::copy_bytes(stream, buffer, actual);
  stream.sync();

  ASSERT_EQUAL(actual.size(), expected.size());

  for (cuda::std::size_t i = 0; i < expected.size(); ++i)
  {
    ASSERT_EQUAL(expected.begin()[i], actual[i]);
  }
}
} // namespace test

#define TEST_ASSERT_DEVICE(condition) ::test::assert_device((condition), #condition, __FILE__, __LINE__)
