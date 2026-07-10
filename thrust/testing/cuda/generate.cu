#include <thrust/execution_policy.h>
#include <thrust/generate.h>

#include <cuda/buffer>
#include <cuda/cccl_runtime_test_helper.cuh>
#include <cuda/launch>
#include <cuda/stream>

#include <unittest/unittest.h>

template <typename T>
struct return_value
{
  T val;

  return_value() = default;
  return_value(T v)
      : val(v)
  {}

  _CCCL_HOST_DEVICE T operator()() const
  {
    return val;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
struct generate_kernel
{
  template <typename ExecutionPolicy, typename Result, typename Function>
  __device__ void operator()(ExecutionPolicy exec, Result result, Function f) const
  {
    thrust::generate(exec, result.begin(), result.end(), f);
  }
};

template <typename T, typename ExecutionPolicy>
void TestGenerateDevice(ExecutionPolicy exec, const size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto result = cuda::make_device_buffer<T>(stream, device, n, cuda::no_init);

  const T value{13};
  const auto f = return_value<T>{value};

  cuda::launch(stream, test_runtime::single_thread_config(), generate_kernel{}, exec, result, f);
  stream.sync();

  test_runtime::assert_filled(stream, result, value);
}

template <typename T>
void TestGenerateDeviceSeq(const size_t n)
{
  TestGenerateDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateDeviceSeq);

template <typename T>
void TestGenerateDeviceDevice(const size_t n)
{
  TestGenerateDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateDeviceDevice);
#endif

void TestGenerateCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto result = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  const int value = 13;
  const auto f    = return_value<int>{value};

  thrust::generate(thrust::cuda::par.on(stream.get()), result.begin(), result.end(), f);
  stream.sync();

  test_runtime::assert_equal(stream, result, {13, 13, 13, 13, 13});
}
DECLARE_UNITTEST(TestGenerateCudaStreams);

#ifdef THRUST_TEST_DEVICE_SIDE
struct generate_n_kernel
{
  template <typename ExecutionPolicy, typename Result, typename Function>
  __device__ void operator()(ExecutionPolicy exec, Result result, Function f) const
  {
    thrust::generate_n(exec, result.begin(), result.size(), f);
  }
};

template <typename T, typename ExecutionPolicy>
void TestGenerateNDevice(ExecutionPolicy exec, const size_t n)
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto result = cuda::make_device_buffer<T>(stream, device, n, cuda::no_init);

  const T value{13};
  const auto f = return_value<T>{value};

  cuda::launch(stream, test_runtime::single_thread_config(), generate_n_kernel{}, exec, result, f);
  stream.sync();

  test_runtime::assert_filled(stream, result, value);
}

template <typename T>
void TestGenerateNDeviceSeq(const size_t n)
{
  TestGenerateNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateNDeviceSeq);

template <typename T>
void TestGenerateNDeviceDevice(const size_t n)
{
  TestGenerateNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGenerateNDeviceDevice);
#endif

void TestGenerateNCudaStreams()
{
  const auto device = test_runtime::current_test_device();
  cuda::stream stream{device};

  auto result = cuda::make_device_buffer<int>(stream, device, 5, cuda::no_init);

  const int value = 13;
  const auto f    = return_value<int>{value};

  thrust::generate_n(thrust::cuda::par.on(stream.get()), result.begin(), result.size(), f);
  stream.sync();

  test_runtime::assert_equal(stream, result, {13, 13, 13, 13, 13});
}
DECLARE_UNITTEST(TestGenerateNCudaStreams);
