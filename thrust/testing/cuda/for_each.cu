#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <algorithm>

#include "thrust/device_vector.h"
#include <unittest/unittest.h>

static const size_t NUM_REGISTERS = 64;

template <size_t N>
_CCCL_HOST_DEVICE void f(int* x)
{
  int temp = *x;
  f<N - 1>(x + 1);
  *x = temp;
};
template <>
_CCCL_HOST_DEVICE void f<0>(int* /*x*/)
{}
template <size_t N>
struct CopyFunctorWithManyRegisters
{
  _CCCL_HOST_DEVICE void operator()(int* ptr)
  {
    f<N>(ptr);
  }
};

void TestForEachLargeRegisterFootprint()
{
  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  thrust::device_vector<int> data(NUM_REGISTERS, 12345);

  thrust::device_vector<int*> input(1, thrust::raw_pointer_cast(&data[0])); // length is irrelevant

  thrust::for_each(input.begin(), input.end(), CopyFunctorWithManyRegisters<NUM_REGISTERS>());
}
DECLARE_UNITTEST(TestForEachLargeRegisterFootprint);

void TestForEachNLargeRegisterFootprint()
{
  int current_device = -1;
  cudaGetDevice(&current_device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  thrust::device_vector<int> data(NUM_REGISTERS, 12345);

  thrust::device_vector<int*> input(1, thrust::raw_pointer_cast(&data[0])); // length is irrelevant

  thrust::for_each_n(input.begin(), input.size(), CopyFunctorWithManyRegisters<NUM_REGISTERS>());
}
DECLARE_UNITTEST(TestForEachNLargeRegisterFootprint);

template <typename T>
struct mark_present_for_each
{
  T* ptr;
  _CCCL_HOST_DEVICE void operator()(T x)
  {
    ptr[(int) x] = 1;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename Function>
__global__ void for_each_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Function f)
{
  thrust::for_each(exec, first, last, f);
}

template <typename T>
void TestForEachDeviceSeq(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = ((size_t) h_input[i]) % output_size;
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  thrust::for_each(h_input.begin(), h_input.end(), h_f);

  for_each_kernel<<<1, 1>>>(thrust::seq, d_input.begin(), d_input.end(), d_f);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachDeviceSeq);

template <typename T>
void TestForEachDeviceDevice(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = ((size_t) h_input[i]) % output_size;
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  thrust::for_each(h_input.begin(), h_input.end(), h_f);

  for_each_kernel<<<1, 1>>>(thrust::device, d_input.begin(), d_input.end(), d_f);
  {
    cudaError_t const err = cudaGetLastError();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachDeviceDevice);

template <typename ExecutionPolicy, typename Iterator, typename Size, typename Function>
__global__ void for_each_n_kernel(ExecutionPolicy exec, Iterator first, Size n, Function f)
{
  thrust::for_each_n(exec, first, n, f);
}

template <typename T>
void TestForEachNDeviceSeq(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = static_cast<T>(((size_t) h_input[i]) % output_size);
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  thrust::for_each_n(h_input.begin(), h_input.size(), h_f);

  for_each_n_kernel<<<1, 1>>>(thrust::seq, d_input.begin(), d_input.size(), d_f);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachNDeviceSeq);

template <typename T>
void TestForEachNDeviceDevice(const size_t n)
{
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<T> h_input = unittest::random_integers<T>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_input[i] = static_cast<T>(((size_t) h_input[i]) % output_size);
  }

  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(output_size, (T) 0);
  thrust::device_vector<T> d_output(output_size, (T) 0);

  mark_present_for_each<T> h_f;
  mark_present_for_each<T> d_f;
  h_f.ptr = &h_output[0];
  d_f.ptr = (&d_output[0]).get();

  thrust::for_each_n(h_input.begin(), h_input.size(), h_f);

  for_each_n_kernel<<<1, 1>>>(thrust::device, d_input.begin(), d_input.size(), d_f);
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_output, d_output);
}
DECLARE_VARIABLE_UNITTEST(TestForEachNDeviceDevice);
#endif

void TestForEachCudaStreams()
{
  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::device_vector<int> input{3, 2, 3, 4, 6};
  thrust::device_vector<int> output(7, 0);

  mark_present_for_each<int> f;
  f.ptr = thrust::raw_pointer_cast(output.data());

  thrust::for_each(thrust::cuda::par.on(s), input.begin(), input.end(), f);

  cudaStreamSynchronize(s);

  thrust::device_vector<int> ref{0, 0, 1, 1, 1, 0, 1};
  ASSERT_EQUAL(output, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestForEachCudaStreams);
