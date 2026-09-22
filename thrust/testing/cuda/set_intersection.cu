#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct set_intersection_kernel
{
  template <typename ExecutionPolicy, typename Input1, typename Input2, typename Output>
  __device__ void operator()(ExecutionPolicy exec, Input1 a, Input2 b, Output result) const
  {
    const auto end = thrust::set_intersection(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    TEST_ASSERT_DEVICE(end == result.end());
  }
};

template <typename ExecutionPolicy>
void TestSetIntersectionDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a      = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto b      = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto result = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);

  cuda::launch(stream, test_runtime::single_thread_config(), set_intersection_kernel{}, exec, a, b, result);
  stream.sync();

  test_runtime::assert_equal(stream, result, {0, 4});
}

TEST_CASE("TestSetIntersectionDeviceSeq", "[set_intersection]")
{
  TestSetIntersectionDevice(thrust::seq);
}

TEST_CASE("TestSetIntersectionDeviceDevice", "[set_intersection]")
{
  TestSetIntersectionDevice(thrust::device);
}

TEST_CASE("TestSetIntersectionDeviceNoSync", "[set_intersection]")
{
  TestSetIntersectionDevice(thrust::cuda::par_nosync);
}
#endif

template <typename ExecutionPolicy>
void TestSetIntersectionCudaStreams(ExecutionPolicy policy)
{
  const auto device = test_runtime::current_test_device();
  const cuda::stream stream{device};

  auto a      = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto b      = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto result = cuda::make_device_buffer<int>(stream, device, 2, cuda::no_init);

  const auto streampolicy = policy.on(stream.get());

  const auto end = thrust::set_intersection(streampolicy, a.begin(), a.end(), b.begin(), b.end(), result.begin());
  stream.sync();

  REQUIRE(result.end() == end);
  test_runtime::assert_equal(stream, result, {0, 4});
}

TEST_CASE("TestSetIntersectionCudaStreamsSync", "[set_intersection]")
{
  TestSetIntersectionCudaStreams(thrust::cuda::par);
}

TEST_CASE("TestSetIntersectionCudaStreamsNoSync", "[set_intersection]")
{
  TestSetIntersectionCudaStreams(thrust::cuda::par_nosync);
}
