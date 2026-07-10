#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct equal_kernel
{
  template <typename ExecutionPolicy, typename Input1, typename Input2>
  __device__ void
  operator()(ExecutionPolicy exec, Input1 data1, Input2 data2, cuda::std::size_t size, bool expected) const
  {
    const auto result = thrust::equal(exec, data1.begin(), data1.begin() + size, data2.begin());
    TEST_ASSERT_DEVICE(result == expected);
  }
};

struct equal_pred_kernel
{
  template <typename ExecutionPolicy, typename Input1, typename Input2, typename BinaryPredicate>
  __device__ void operator()(
    ExecutionPolicy exec, Input1 data1, Input2 data2, cuda::std::size_t size, BinaryPredicate pred, bool expected) const
  {
    const auto result = thrust::equal(exec, data1.begin(), data1.begin() + size, data2.begin(), pred);
    TEST_ASSERT_DEVICE(result == expected);
  }
};

template <typename T, typename ExecutionPolicy>
void TestEqualDevice(ExecutionPolicy exec, const size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto h_data1 = test_runtime::random_samples_buffer<T>(stream, n);
  auto h_data2 = test_runtime::random_samples_buffer<T>(stream, n, n);

  if (n > 0)
  {
    h_data1[0] = T{0};
    h_data2[0] = T{1};
  }

  auto d_data1 = cuda::make_device_buffer<T>(stream, device, h_data1);
  auto d_data2 = cuda::make_device_buffer<T>(stream, device, h_data2);

  // empty ranges
  cuda::launch(stream, test_runtime::single_thread_config(), equal_kernel{}, exec, d_data1, d_data1, 0, true);
  stream.sync();

  // symmetric cases
  cuda::launch(stream, test_runtime::single_thread_config(), equal_kernel{}, exec, d_data1, d_data1, n, true);
  stream.sync();

  if (n > 0)
  {
    // different vectors
    cuda::launch(stream, test_runtime::single_thread_config(), equal_kernel{}, exec, d_data1, d_data2, n, false);
    stream.sync();

    // different predicates
    cuda::launch(
      stream,
      test_runtime::single_thread_config(),
      equal_pred_kernel{},
      exec,
      d_data1,
      d_data2,
      1,
      ::cuda::std::less<T>{},
      true);
    stream.sync();

    cuda::launch(
      stream,
      test_runtime::single_thread_config(),
      equal_pred_kernel{},
      exec,
      d_data1,
      d_data2,
      1,
      ::cuda::std::greater<T>{},
      false);
    stream.sync();
  }
}

template <typename T>
void TestEqualDeviceSeq(const size_t n)
{
  TestEqualDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceSeq);

template <typename T>
void TestEqualDeviceDevice(const size_t n)
{
  TestEqualDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestEqualDeviceDevice);
#endif

void TestEqualCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v1     = cuda::make_device_buffer<int>(stream, device, {5, 2, 0, 0, 0});
  auto v2     = cuda::make_device_buffer<int>(stream, device, {5, 2, 0, 6, 1});
  auto policy = thrust::cuda::par.on(stream.get());

  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.end(), v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.end(), v2.begin()), false);
  ASSERT_EQUAL(thrust::equal(policy, v2.begin(), v2.end(), v2.begin()), true);

  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.begin() + 0, v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.begin() + 1, v1.begin()), true);
  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.begin() + 3, v2.begin()), true);
  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.begin() + 4, v2.begin()), false);

  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.end(), v2.begin(), ::cuda::std::less_equal<int>{}), true);
  ASSERT_EQUAL(thrust::equal(policy, v1.begin(), v1.end(), v2.begin(), ::cuda::std::greater<int>{}), false);

  stream.sync();
}
DECLARE_UNITTEST(TestEqualCudaStreams);
