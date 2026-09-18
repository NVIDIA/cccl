#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct set_intersection_by_key_kernel
{
  template <typename ExecutionPolicy,
            typename Keys1,
            typename Keys2,
            typename Values1,
            typename KeysOutput,
            typename ValuesOutput>
  __device__ void operator()(
    ExecutionPolicy exec, Keys1 keys1, Keys2 keys2, Values1 values1, KeysOutput keys_result, ValuesOutput values_result)
    const
  {
    auto end = thrust::set_intersection_by_key(
      exec,
      keys1.begin(),
      keys1.end(),
      keys2.begin(),
      keys2.end(),
      values1.begin(),
      keys_result.begin(),
      values_result.begin());
    TEST_ASSERT_DEVICE(end.first == keys_result.end());
    TEST_ASSERT_DEVICE(end.second == values_result.end());
  }
};

template <typename ExecutionPolicy>
void TestSetIntersectionByKeyDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a_key      = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto b_key      = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto a_val      = cuda::make_device_buffer<int>(stream, device, {0, 0, 0});
  auto result_key = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);
  auto result_val = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    set_intersection_by_key_kernel{},
    exec,
    a_key,
    b_key,
    a_val,
    result_key,
    result_val);
  stream.sync();

  test_runtime::assert_equal(stream, result_key, {0, 4});
  test_runtime::assert_equal(stream, result_val, {0, 0});
}

void TestSetIntersectionByKeyDeviceSeq()
{
  TestSetIntersectionByKeyDevice(thrust::seq);
}
TEST_CASE("TestSetIntersectionByKeyDeviceSeq", "[set_intersection_by_key]")
{
  TestSetIntersectionByKeyDeviceSeq();
}

void TestSetIntersectionByKeyDeviceDevice()
{
  TestSetIntersectionByKeyDevice(thrust::device);
}
TEST_CASE("TestSetIntersectionByKeyDeviceDevice", "[set_intersection_by_key]")
{
  TestSetIntersectionByKeyDeviceDevice();
}

void TestSetIntersectionByKeyDeviceNoSync()
{
  TestSetIntersectionByKeyDevice(thrust::cuda::par_nosync);
}
TEST_CASE("TestSetIntersectionByKeyDeviceNoSync", "[set_intersection_by_key]")
{
  TestSetIntersectionByKeyDeviceNoSync();
}
#endif

template <typename ExecutionPolicy>
void TestSetIntersectionByKeyCudaStreams(ExecutionPolicy policy)
{
  const auto device = test_runtime::current_test_device();
  const cuda::stream stream{device};

  auto a_key      = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto b_key      = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto a_val      = cuda::make_device_buffer<int>(stream, device, {0, 0, 0});
  auto result_key = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);
  auto result_val = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);

  const auto streampolicy = policy.on(stream.get());

  const auto end = thrust::set_intersection_by_key(
    streampolicy,
    a_key.begin(),
    a_key.end(),
    b_key.begin(),
    b_key.end(),
    a_val.begin(),
    result_key.begin(),
    result_val.begin());
  stream.sync();

  REQUIRE(result_key.end() == end.first);
  REQUIRE(result_val.end() == end.second);
  test_runtime::assert_equal(stream, result_key, {0, 4});
  test_runtime::assert_equal(stream, result_val, {0, 0});
}

void TestSetIntersectionByKeyCudaStreamsSync()
{
  TestSetIntersectionByKeyCudaStreams(thrust::cuda::par);
}
TEST_CASE("TestSetIntersectionByKeyCudaStreamsSync", "[set_intersection_by_key]")
{
  TestSetIntersectionByKeyCudaStreamsSync();
}

void TestSetIntersectionByKeyCudaStreamsNoSync()
{
  TestSetIntersectionByKeyCudaStreams(thrust::cuda::par_nosync);
}
TEST_CASE("TestSetIntersectionByKeyCudaStreamsNoSync", "[set_intersection_by_key]")
{
  TestSetIntersectionByKeyCudaStreamsNoSync();
}
