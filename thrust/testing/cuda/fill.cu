#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <algorithm>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct fill_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void
  operator()(ExecutionPolicy exec, Data data, cuda::std::size_t first, cuda::std::size_t last, T value) const
  {
    thrust::fill(exec, data.begin() + first, data.begin() + last, value);
  }
};

template <typename T, typename ExecutionPolicy>
void TestFillDevice(ExecutionPolicy exec, size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto expected = test_runtime::random_integers_buffer<T>(stream, n);
  auto data     = cuda::make_device_buffer<T>(stream, device, expected);

  const auto run_fill = [&](size_t first, size_t last, T value) {
    for (size_t i = first; i < last; ++i)
    {
      expected[i] = value;
    }

    cuda::launch(stream, test_runtime::single_thread_config(), fill_kernel{}, exec, data, first, last, value);
    stream.sync();

    test_runtime::assert_equal(stream, data, expected);
  };

  run_fill(std::min<size_t>(1, n), std::min<size_t>(3, n), T{0});
  run_fill(std::min<size_t>(117, n), std::min<size_t>(367, n), T{1});
  run_fill(std::min<size_t>(8, n), std::min<size_t>(259, n), T{2});
  run_fill(std::min<size_t>(3, n), n, T{3});
  run_fill(0, n, T{4});
}

template <typename T>
void TestFillDeviceSeq(size_t n)
{
  TestFillDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceSeq);

template <typename T>
void TestFillDeviceDevice(size_t n)
{
  TestFillDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceDevice);

struct fill_n_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void
  operator()(ExecutionPolicy exec, Data data, cuda::std::size_t first, cuda::std::size_t count, T value) const
  {
    thrust::fill_n(exec, data.begin() + first, count, value);
  }
};

template <typename T, typename ExecutionPolicy>
void TestFillNDevice(ExecutionPolicy exec, size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto expected = test_runtime::random_integers_buffer<T>(stream, n);
  auto data     = cuda::make_device_buffer<T>(stream, device, expected);

  const auto run_fill_n = [&](size_t first, size_t count, T value) {
    for (size_t i = first; i < first + count; ++i)
    {
      expected[i] = value;
    }

    cuda::launch(stream, test_runtime::single_thread_config(), fill_n_kernel{}, exec, data, first, count, value);
    stream.sync();

    test_runtime::assert_equal(stream, data, expected);
  };

  size_t first = std::min<size_t>(1, n);
  run_fill_n(first, std::min<size_t>(3, n) - first, T{0});

  first = std::min<size_t>(117, n);
  run_fill_n(first, std::min<size_t>(367, n) - first, T{1});

  first = std::min<size_t>(8, n);
  run_fill_n(first, std::min<size_t>(259, n) - first, T{2});

  first = std::min<size_t>(3, n);
  run_fill_n(first, n - first, T{3});

  run_fill_n(0, n, T{4});
}

template <typename T>
void TestFillNDeviceSeq(size_t n)
{
  TestFillNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceSeq);

template <typename T>
void TestFillNDeviceDevice(size_t n)
{
  TestFillNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceDevice);
#endif

void TestFillCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, {0, 1, 2, 3, 4});

  thrust::fill(thrust::cuda::par.on(stream.get()), v.begin() + 1, v.begin() + 4, 7);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 7, 7, 7, 4});

  thrust::fill(thrust::cuda::par.on(stream.get()), v.begin(), v.begin() + 3, 8);
  stream.sync();

  test_runtime::assert_equal(stream, v, {8, 8, 8, 7, 4});

  thrust::fill(thrust::cuda::par.on(stream.get()), v.begin() + 2, v.end(), 9);
  stream.sync();

  test_runtime::assert_equal(stream, v, {8, 8, 9, 9, 9});

  thrust::fill(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), 1);
  stream.sync();

  test_runtime::assert_equal(stream, v, {1, 1, 1, 1, 1});
}
DECLARE_UNITTEST(TestFillCudaStreams);
