#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/tabulate.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/std/functional>
#include <cuda/stream>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
struct tabulate_kernel
{
  template <typename ExecutionPolicy, typename Result, typename Function>
  __device__ void operator()(ExecutionPolicy exec, Result result, Function f) const
  {
    thrust::tabulate(exec, result.begin(), result.end(), f);
  }
};

template <typename ExecutionPolicy>
void TestTabulateDevice(ExecutionPolicy exec)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  cuda::launch(stream, test_runtime::single_thread_config(), tabulate_kernel{}, exec, v, ::cuda::std::identity{});
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 2, 3, 4});

  cuda::launch(stream, test_runtime::single_thread_config(), tabulate_kernel{}, exec, v, -thrust::placeholders::_1);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, -1, -2, -3, -4});

  cuda::launch(
    stream,
    test_runtime::single_thread_config(),
    tabulate_kernel{},
    exec,
    v,
    thrust::placeholders::_1 * thrust::placeholders::_1 * thrust::placeholders::_1);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 8, 27, 64});
}

void TestTabulateDeviceSeq()
{
  TestTabulateDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTabulateDeviceSeq);

void TestTabulateDeviceDevice()
{
  TestTabulateDevice(thrust::device);
}
DECLARE_UNITTEST(TestTabulateDeviceDevice);
#endif

void TestTabulateCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto v = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  thrust::tabulate(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), ::cuda::std::identity{});
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 2, 3, 4});

  thrust::tabulate(thrust::cuda::par.on(stream.get()), v.begin(), v.end(), -thrust::placeholders::_1);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, -1, -2, -3, -4});

  thrust::tabulate(thrust::cuda::par.on(stream.get()),
                   v.begin(),
                   v.end(),
                   thrust::placeholders::_1 * thrust::placeholders::_1 * thrust::placeholders::_1);
  stream.sync();

  test_runtime::assert_equal(stream, v, {0, 1, 8, 27, 64});
}
DECLARE_UNITTEST(TestTabulateCudaStreams);
