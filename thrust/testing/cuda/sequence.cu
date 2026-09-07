#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct sequence_kernel
{
  template <typename ExecutionPolicy, typename Data>
  __device__ void operator()(ExecutionPolicy exec, Data data) const
  {
    thrust::sequence(exec, data.begin(), data.end());
  }
};

struct sequence_init_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void operator()(ExecutionPolicy exec, Data data, T init) const
  {
    thrust::sequence(exec, data.begin(), data.end(), init);
  }
};

struct sequence_init_step_kernel
{
  template <typename ExecutionPolicy, typename Data, typename T>
  __device__ void operator()(ExecutionPolicy exec, Data data, T init, T step) const
  {
    thrust::sequence(exec, data.begin(), data.end(), init, step);
  }
};

template <typename ExecutionPolicy>
void TestSequenceDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  cuda::launch(stream, test_runtime::single_thread_config(), sequence_kernel{}, exec, v);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 2, 3, 4});

  cuda::launch(stream, test_runtime::single_thread_config(), sequence_init_kernel{}, exec, v, 10);
  stream.sync();

  test_runtime::assert_equal(stream, v, {10, 11, 12, 13, 14});

  cuda::launch(stream, test_runtime::single_thread_config(), sequence_init_step_kernel{}, exec, v, 10, 2);
  stream.sync();

  test_runtime::assert_equal(stream, v, {10, 12, 14, 16, 18});
}

void TestSequenceDeviceSeq()
{
  TestSequenceDevice(thrust::seq);
}
DECLARE_UNITTEST(TestSequenceDeviceSeq);

void TestSequenceDeviceDevice()
{
  TestSequenceDevice(thrust::device);
}
DECLARE_UNITTEST(TestSequenceDeviceDevice);
#endif

void TestSequenceCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  thrust::sequence(thrust::cuda::par.on(stream.get()), v.begin(), v.end());
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 2, 3, 4});

  thrust::sequence(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), 10);
  stream.sync();

  test_runtime::assert_equal(stream, v, {10, 11, 12, 13, 14});

  thrust::sequence(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), 10, 2);
  stream.sync();

  test_runtime::assert_equal(stream, v, {10, 12, 14, 16, 18});
}
DECLARE_UNITTEST(TestSequenceCudaStreams);
