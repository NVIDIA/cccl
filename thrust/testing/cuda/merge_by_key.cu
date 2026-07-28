#include <thrust/execution_policy.h>
#include <thrust/merge.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct merge_by_key_kernel
{
  template <typename ExecutionPolicy,
            typename Keys1,
            typename Keys2,
            typename Values1,
            typename Values2,
            typename KeysOutput,
            typename ValuesOutput>
  __device__ void operator()(
    ExecutionPolicy exec,
    Keys1 keys1,
    Keys2 keys2,
    Values1 values1,
    Values2 values2,
    KeysOutput keys_result,
    ValuesOutput values_result) const
  {
    auto end = thrust::merge_by_key(
      exec,
      keys1.begin(),
      keys1.end(),
      keys2.begin(),
      keys2.end(),
      values1.begin(),
      values2.begin(),
      keys_result.begin(),
      values_result.begin());
    TEST_ASSERT_DEVICE(end.first == keys_result.end());
    TEST_ASSERT_DEVICE(end.second == values_result.end());
  }
};

template <typename ExecutionPolicy>
void TestMergeByKeyDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a_key = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto a_val = cuda::make_device_buffer<int>(stream, device, {13, 7, 42});
  auto b_key = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto b_val = cuda::make_device_buffer<int>(stream, device, {42, 42, 7, 13});

  auto result_key = cuda::make_device_buffer<int>(stream, device, 7, cuda::no_init);
  auto result_val = cuda::make_device_buffer<int>(stream, device, 7, cuda::no_init);

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    merge_by_key_kernel{},
    exec,
    a_key,
    b_key,
    a_val,
    b_val,
    result_key,
    result_val);
  stream.sync();

  test_runtime::assert_equal(stream, result_key, {0, 0, 2, 3, 3, 4, 4});
  test_runtime::assert_equal(stream, result_val, {13, 42, 7, 42, 7, 42, 13});
}

void TestMergeByKeyDeviceSeq()
{
  TestMergeByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMergeByKeyDeviceSeq);

void TestMergeByKeyDeviceDevice()
{
  TestMergeByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestMergeByKeyDeviceDevice);
#endif

void TestMergeByKeyCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a_key = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto a_val = cuda::make_device_buffer<int>(stream, device, {13, 7, 42});
  auto b_key = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto b_val = cuda::make_device_buffer<int>(stream, device, {42, 42, 7, 13});

  auto result_key = cuda::make_device_buffer<int>(stream, device, 7, cuda::no_init);
  auto result_val = cuda::make_device_buffer<int>(stream, device, 7, cuda::no_init);

  const auto end = thrust::merge_by_key(
    thrust::cuda::par.on(stream.get()),
    a_key.begin(),
    a_key.end(),
    b_key.begin(),
    b_key.end(),
    a_val.begin(),
    b_val.begin(),
    result_key.begin(),
    result_val.begin());
  stream.sync();

  ASSERT_EQUAL_QUIET(result_key.end(), end.first);
  ASSERT_EQUAL_QUIET(result_val.end(), end.second);
  test_runtime::assert_equal(stream, result_key, {0, 0, 2, 3, 3, 4, 4});
  test_runtime::assert_equal(stream, result_val, {13, 42, 7, 42, 7, 42, 13});
}
DECLARE_UNITTEST(TestMergeByKeyCudaStreams);
