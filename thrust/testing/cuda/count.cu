#include <thrust/count.h>
#include <thrust/execution_policy.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct count_kernel
{
  template <typename ExecutionPolicy, typename Input, typename T, typename Expected>
  __device__ void operator()(ExecutionPolicy exec, Input data, T value, Expected expected) const
  {
    const auto result = thrust::count(exec, data.begin(), data.end(), value);
    TEST_ASSERT_DEVICE(result == expected);
  }
};

template <typename T, typename ExecutionPolicy>
void TestCountDevice(ExecutionPolicy exec, const size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto h_data = test_runtime::random_samples_buffer<T>(stream, n);
  auto d_data = cuda::make_device_buffer<T>(stream, device, h_data);

  const auto expected = thrust::count(h_data.begin(), h_data.end(), T{5});

  cuda::launch(stream, test_runtime::single_thread_config(), count_kernel{}, exec, d_data, T{5}, expected);
  stream.sync();
}

template <typename T>
void TestCountDeviceSeq(const size_t n)
{
  TestCountDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountDeviceSeq);

template <typename T>
void TestCountDeviceDevice(const size_t n)
{
  TestCountDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountDeviceDevice);

template <typename T>
struct greater_than_five
{
  _CCCL_HOST_DEVICE bool operator()(const T& x) const
  {
    return x > 5;
  }
};

struct count_if_kernel
{
  template <typename ExecutionPolicy, typename Input, typename Predicate, typename Expected>
  __device__ void operator()(ExecutionPolicy exec, Input data, Predicate pred, Expected expected) const
  {
    const auto result = thrust::count_if(exec, data.begin(), data.end(), pred);
    TEST_ASSERT_DEVICE(result == expected);
  }
};

template <typename T, typename ExecutionPolicy>
void TestCountIfDevice(ExecutionPolicy exec, const size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto h_data = test_runtime::random_samples_buffer<T>(stream, n);
  auto d_data = cuda::make_device_buffer<T>(stream, device, h_data);

  const auto expected = thrust::count_if(h_data.begin(), h_data.end(), greater_than_five<T>{});

  cuda::launch(
    stream, test_runtime::single_thread_config(), count_if_kernel{}, exec, d_data, greater_than_five<T>{}, expected);
  stream.sync();
}

template <typename T>
void TestCountIfDeviceSeq(const size_t n)
{
  TestCountIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountIfDeviceSeq);

template <typename T>
void TestCountIfDeviceDevice(const size_t n)
{
  TestCountIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestCountIfDeviceDevice);
#endif

void TestCountCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto data   = cuda::make_device_buffer<int>(stream, device, {1, 1, 0, 0, 1});
  auto policy = thrust::cuda::par.on(stream.get());

  ASSERT_EQUAL(thrust::count(policy, data.begin(), data.end(), 0), 2);
  ASSERT_EQUAL(thrust::count(policy, data.begin(), data.end(), 1), 3);
  ASSERT_EQUAL(thrust::count(policy, data.begin(), data.end(), 2), 0);

  stream.sync();
}
DECLARE_UNITTEST(TestCountCudaStreams);
