#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct max_element_kernel
{
  template <typename ExecutionPolicy, typename Input>
  __device__ void operator()(ExecutionPolicy exec, Input data, cuda::std::ptrdiff_t expected) const
  {
    const auto result = thrust::max_element(exec, data.begin(), data.end());
    TEST_ASSERT_DEVICE(result - data.begin() == expected);
  }
};

struct max_element_pred_kernel
{
  template <typename ExecutionPolicy, typename Input, typename BinaryPredicate>
  __device__ void operator()(ExecutionPolicy exec, Input data, BinaryPredicate pred, cuda::std::ptrdiff_t expected) const
  {
    const auto result = thrust::max_element(exec, data.begin(), data.end(), pred);
    TEST_ASSERT_DEVICE(result - data.begin() == expected);
  }
};

template <typename ExecutionPolicy>
void TestMaxElementDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  constexpr cuda::std::size_t n = 1000;
  auto h_data                   = test_runtime::random_samples_buffer<int>(stream, n);
  auto d_data                   = cuda::make_device_buffer<int>(stream, device, h_data);

  const auto h_max = thrust::max_element(h_data.begin(), h_data.end());

  cuda::launch(stream, test_runtime::single_thread_config(), max_element_kernel{}, exec, d_data, h_max - h_data.begin());
  stream.sync();

  const auto h_min = thrust::max_element(h_data.begin(), h_data.end(), ::cuda::std::greater<int>{});

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    max_element_pred_kernel{},
    exec,
    d_data,
    ::cuda::std::greater<int>{},
    h_min - h_data.begin());
  stream.sync();
}

void TestMaxElementDeviceSeq()
{
  TestMaxElementDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMaxElementDeviceSeq);

void TestMaxElementDeviceDevice()
{
  TestMaxElementDevice(thrust::device);
}
DECLARE_UNITTEST(TestMaxElementDeviceDevice);

void TestMaxElementDeviceNoSync()
{
  TestMaxElementDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestMaxElementDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestMaxElementCudaStreams(ExecutionPolicy policy)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto data = cuda::make_device_buffer<int>(stream, device, {3, 5, 1, 2, 5, 1});

  auto streampolicy = policy.on(stream.get());

  ASSERT_EQUAL(thrust::max_element(streampolicy, data.begin(), data.end()) - data.begin(), 1);

  ASSERT_EQUAL(thrust::max_element(streampolicy, data.begin(), data.end(), ::cuda::std::greater<int>{}) - data.begin(),
               2);

  stream.sync();
}

void TestMaxElementCudaStreamsSync()
{
  TestMaxElementCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestMaxElementCudaStreamsSync);

void TestMaxElementCudaStreamsNoSync()
{
  TestMaxElementCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestMaxElementCudaStreamsNoSync);

void TestMaxElementDevicePointer()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto data = cuda::make_device_buffer<int>(stream, device, {3, 5, 1, 2, 5, 1});

  auto policy   = thrust::cuda::par.on(stream.get());
  auto* raw_ptr = data.data();
  const auto n  = data.size();
  ASSERT_EQUAL(thrust::max_element(policy, raw_ptr, raw_ptr + n) - raw_ptr, 1);
  ASSERT_EQUAL(thrust::max_element(policy, raw_ptr, raw_ptr + n, ::cuda::std::greater<int>{}) - raw_ptr, 2);

  stream.sync();
}
DECLARE_UNITTEST(TestMaxElementDevicePointer);
