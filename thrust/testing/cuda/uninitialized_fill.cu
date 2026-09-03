#include <thrust/execution_policy.h>
#include <thrust/uninitialized_fill.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct uninitialized_fill_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void
  operator()(ExecutionPolicy exec, Data data, cuda::std::size_t first, cuda::std::size_t last, T value) const
  {
    thrust::uninitialized_fill(exec, data.begin() + first, data.begin() + last, value);
  }
};

template <typename ExecutionPolicy>
void TestUninitializedFillDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, {0, 1, 2, 3, 4});

  int value = 7;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_kernel{}, exec, v, 1, 4, value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, value, value, value, 4});

  value = 8;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_kernel{}, exec, v, 0, 3, value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, 7, 4});

  value = 9;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_kernel{}, exec, v, 2, v.size(), value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {8, 8, value, value, 9});

  value = 1;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_kernel{}, exec, v, 0, v.size(), value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, value, value});
}

void TestUninitializedFillDeviceSeq()
{
  TestUninitializedFillDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceSeq);

void TestUninitializedFillDeviceDevice()
{
  TestUninitializedFillDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillDeviceDevice);
#endif

void TestUninitializedFillCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v          = cuda::make_device_buffer<int>(stream, device, {0, 1, 2, 3, 4});
  const int value = 7;

  thrust::uninitialized_fill(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, value, value});
}
DECLARE_UNITTEST(TestUninitializedFillCudaStreams);

#ifdef THRUST_TEST_DEVICE_SIDE
struct uninitialized_fill_n_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void
  operator()(ExecutionPolicy exec, Data data, cuda::std::size_t first, cuda::std::size_t count, T value) const
  {
    const auto iter = thrust::uninitialized_fill_n(exec, data.begin() + first, count, value);
    TEST_ASSERT_DEVICE(iter == data.begin() + first + count);
  }
};

template <typename ExecutionPolicy>
void TestUninitializedFillNDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, {0, 1, 2, 3, 4});

  int value = 7;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_n_kernel{}, exec, v, 1, 3, value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, value, value, value, 4});

  value = 8;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_n_kernel{}, exec, v, 0, 3, value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, 7, 4});

  value = 9;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_n_kernel{}, exec, v, 2, 3, value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {8, 8, value, value, 9});

  value = 1;
  cuda::launch(stream, test_runtime::single_thread_config(), uninitialized_fill_n_kernel{}, exec, v, 0, v.size(), value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, value, value});
}

void TestUninitializedFillNDeviceSeq()
{
  TestUninitializedFillNDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceSeq);

void TestUninitializedFillNDeviceDevice()
{
  TestUninitializedFillNDevice(thrust::device);
}
DECLARE_UNITTEST(TestUninitializedFillNDeviceDevice);
#endif

void TestUninitializedFillNCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v          = cuda::make_device_buffer<int>(stream, device, {0, 1, 2, 3, 4});
  const int value = 7;

  thrust::uninitialized_fill_n(thrust::cuda::par.on(stream.get()), v.begin(), v.size(), value);
  stream.sync();

  test_runtime::assert_equal(stream, v, {value, value, value, value, value});
}
DECLARE_UNITTEST(TestUninitializedFillNCudaStreams);
