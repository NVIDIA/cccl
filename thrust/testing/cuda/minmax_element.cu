#include <thrust/extrema.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/functional>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct minmax_element_kernel
{
  template <typename ExecutionPolicy, typename Input>
  __device__ void operator()(
    ExecutionPolicy exec, Input data, cuda::std::ptrdiff_t expected_first, cuda::std::ptrdiff_t expected_second) const
  {
    const auto result = thrust::minmax_element(exec, data.begin(), data.end());
    TEST_ASSERT_DEVICE(result.first - data.begin() == expected_first);
    TEST_ASSERT_DEVICE(result.second - data.begin() == expected_second);
  }
};

struct minmax_element_pred_kernel
{
  template <typename ExecutionPolicy, typename Input, typename BinaryPredicate>
  __device__ void operator()(
    ExecutionPolicy exec,
    Input data,
    BinaryPredicate pred,
    cuda::std::ptrdiff_t expected_first,
    cuda::std::ptrdiff_t expected_second) const
  {
    const auto result = thrust::minmax_element(exec, data.begin(), data.end(), pred);
    TEST_ASSERT_DEVICE(result.first - data.begin() == expected_first);
    TEST_ASSERT_DEVICE(result.second - data.begin() == expected_second);
  }
};

template <typename ExecutionPolicy>
void TestMinMaxElementDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  constexpr cuda::std::size_t n = 1000;
  auto h_data                   = test_runtime::random_samples_buffer<int>(stream, n);
  auto d_data                   = cuda::make_device_buffer<int>(stream, device, h_data);

  const auto h_result = thrust::minmax_element(h_data.begin(), h_data.end());

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    minmax_element_kernel{},
    exec,
    d_data,
    h_result.first - h_data.begin(),
    h_result.second - h_data.begin());
  stream.sync();

  const auto h_greater_result = thrust::minmax_element(h_data.begin(), h_data.end(), ::cuda::std::greater<int>{});

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    minmax_element_pred_kernel{},
    exec,
    d_data,
    ::cuda::std::greater<int>{},
    h_greater_result.first - h_data.begin(),
    h_greater_result.second - h_data.begin());
  stream.sync();
}

void TestMinMaxElementDeviceSeq()
{
  TestMinMaxElementDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMinMaxElementDeviceSeq);

void TestMinMaxElementDeviceDevice()
{
  TestMinMaxElementDevice(thrust::device);
}
DECLARE_UNITTEST(TestMinMaxElementDeviceDevice);
#endif

void TestMinMaxElementCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto data = cuda::make_device_buffer<int>(stream, device, {3, 5, 1, 2, 5, 1});

  const auto result = thrust::minmax_element(thrust::cuda::par.on(stream.get()), data.begin(), data.end());

  ASSERT_EQUAL(result.first - data.begin(), 2);
  ASSERT_EQUAL(result.second - data.begin(), 1);

  stream.sync();
}
DECLARE_UNITTEST(TestMinMaxElementCudaStreams);

void TestMinMaxElementDevicePointer()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto data = cuda::make_device_buffer<int>(stream, device, {3, 5, 1, 2, 5, 1});

  auto policy   = thrust::cuda::par.on(stream.get());
  auto* raw_ptr = data.data();
  const auto n  = data.size();
  ASSERT_EQUAL(thrust::minmax_element(policy, raw_ptr, raw_ptr + n).first - raw_ptr, 2);
  ASSERT_EQUAL(thrust::minmax_element(policy, raw_ptr, raw_ptr + n).second - raw_ptr, 1);

  stream.sync();
}
DECLARE_UNITTEST(TestMinMaxElementDevicePointer);
