#include <thrust/execution_policy.h>
#include <thrust/set_operations.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/initializer_list>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct set_union_kernel
{
  template <typename ExecutionPolicy, typename Input1, typename Input2, typename Output>
  __device__ void operator()(ExecutionPolicy exec, Input1 a, Input2 b, Output result) const
  {
    const auto end = thrust::set_union(exec, a.begin(), a.end(), b.begin(), b.end(), result.begin());
    TEST_ASSERT_DEVICE(end == result.end());
  }
};

template <typename ExecutionPolicy>
void TestSetUnionDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a      = cuda::make_device_buffer<int>(stream, device, cuda::std::initializer_list<int>{0, 2, 4});
  auto b      = cuda::make_device_buffer<int>(stream, device, cuda::std::initializer_list<int>{0, 3, 3, 4});
  auto result = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  cuda::launch(stream, test_runtime::single_thread_config(), set_union_kernel{}, exec, a, b, result);
  stream.sync();

  test_runtime::assert_equal(stream, result, {0, 2, 3, 3, 4});
}

void TestSetUnionDeviceSeq()
{
  TestSetUnionDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSetUnionDeviceSeq);

void TestSetUnionDeviceDevice()
{
  TestSetUnionDevice(thrust::device);
}
DECLARE_UNITTEST(TestSetUnionDeviceDevice);
#endif

void TestSetUnionCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a      = cuda::make_device_buffer<int>(stream, device, cuda::std::initializer_list<int>{0, 2, 4});
  auto b      = cuda::make_device_buffer<int>(stream, device, cuda::std::initializer_list<int>{0, 3, 3, 4});
  auto result = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  auto end =
    thrust::set_union(thrust::cuda::par.on(stream.get()), a.begin(), a.end(), b.begin(), b.end(), result.begin());

  ASSERT_EQUAL_QUIET(result.end(), end);
  test_runtime::assert_equal(stream, result, {0, 2, 3, 3, 4});
}
DECLARE_UNITTEST(TestSetUnionCudaStreams);
