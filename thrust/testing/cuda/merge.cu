#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/merge.h>
#include <thrust/sort.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct merge_kernel
{
  template <typename ExecutionPolicy, typename Input1, typename Input2, typename Size, typename Output>
  __device__ void operator()(ExecutionPolicy exec, Input1 a, Input2 b, Size b_size, Output result) const
  {
    const auto end = thrust::merge(exec, a.begin(), a.end(), b.begin(), b.begin() + b_size, result.begin());
    TEST_ASSERT_DEVICE(end == result.end());
  }
};

template <typename ExecutionPolicy>
void TestMergeDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  const size_t n         = 10000;
  const size_t sizes[]   = {0, 1, n / 2, n, n + 1, 2 * n};
  const size_t num_sizes = sizeof(sizes) / sizeof(size_t);

  const auto max_size = static_cast<size_t>(*thrust::max_element(sizes, sizes + num_sizes));
  auto h_a            = test_runtime::random_integers_buffer<int, unittest::int8_t>(stream, n);
  auto h_b            = test_runtime::random_integers_buffer<int, unittest::int8_t>(stream, max_size, n);

  thrust::stable_sort(h_a.begin(), h_a.end());
  thrust::stable_sort(h_b.begin(), h_b.end());

  auto d_a = cuda::make_device_buffer<int>(stream, device, h_a);
  auto d_b = cuda::make_device_buffer<int>(stream, device, h_b);

  for (size_t i = 0; i < num_sizes; i++)
  {
    const size_t size = sizes[i];

    auto h_result = test_runtime::make_host_buffer<int>(stream, n + size);
    stream.sync();

    const auto h_end = thrust::merge(h_a.begin(), h_a.end(), h_b.begin(), h_b.begin() + size, h_result.begin());
    ASSERT_EQUAL_QUIET(h_result.end(), h_end);

    auto result = cuda::make_device_buffer<int>(stream, device, h_result.size(), cuda::no_init);

    cuda::launch(stream, test_runtime::single_thread_config(), merge_kernel{}, exec, d_a, d_b, size, result);
    stream.sync();

    test_runtime::assert_equal(stream, result, h_result);
  }
}

void TestMergeDeviceSeq()
{
  TestMergeDevice(thrust::seq);
}
DECLARE_UNITTEST(TestMergeDeviceSeq);

void TestMergeDeviceDevice()
{
  TestMergeDevice(thrust::device);
}
DECLARE_UNITTEST(TestMergeDeviceDevice);
#endif

void TestMergeCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto a      = cuda::make_device_buffer<int>(stream, device, {0, 2, 4});
  auto b      = cuda::make_device_buffer<int>(stream, device, {0, 3, 3, 4});
  auto result = cuda::make_device_buffer<int>(stream, device, 7, cuda::no_init);

  const auto end =
    thrust::merge(thrust::cuda::par.on(stream.get()), a.begin(), a.end(), b.begin(), b.end(), result.begin());
  stream.sync();

  ASSERT_EQUAL_QUIET(result.end(), end);
  test_runtime::assert_equal(stream, result, {0, 0, 2, 3, 3, 4, 4});
}
DECLARE_UNITTEST(TestMergeCudaStreams);
