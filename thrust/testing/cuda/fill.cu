#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename T>
__global__ void fill_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T value)
{
  thrust::fill(exec, first, last, value);
}

template <typename T, typename ExecutionPolicy>
void TestFillDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::fill(h_data.begin() + std::min((size_t) 1, n), h_data.begin() + std::min((size_t) 3, n), (T) 0);

  fill_kernel<<<1, 1>>>(exec, d_data.begin() + std::min((size_t) 1, n), d_data.begin() + std::min((size_t) 3, n), (T) 0);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 117, n), h_data.begin() + std::min((size_t) 367, n), (T) 1);

  fill_kernel<<<1, 1>>>(
    exec, d_data.begin() + std::min((size_t) 117, n), d_data.begin() + std::min((size_t) 367, n), (T) 1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 8, n), h_data.begin() + std::min((size_t) 259, n), (T) 2);

  fill_kernel<<<1, 1>>>(
    exec, d_data.begin() + std::min((size_t) 8, n), d_data.begin() + std::min((size_t) 259, n), (T) 2);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 3, n), h_data.end(), (T) 3);

  fill_kernel<<<1, 1>>>(exec, d_data.begin() + std::min((size_t) 3, n), d_data.end(), (T) 3);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  thrust::fill(h_data.begin(), h_data.end(), (T) 4);

  fill_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), (T) 4);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);
}

template <typename T>
void TestFillDeviceSeq(size_t n)
{
  TestFillDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceSeq);

template <typename T>
void TestFillDeviceDevice(size_t n)
{
  TestFillDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillDeviceDevice);

template <typename ExecutionPolicy, typename Iterator, typename Size, typename T>
__global__ void fill_n_kernel(ExecutionPolicy exec, Iterator first, Size n, T value)
{
  thrust::fill_n(exec, first, n, value);
}

template <typename T, typename ExecutionPolicy>
void TestFillNDevice(ExecutionPolicy exec, size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  size_t begin_offset = std::min<size_t>(1, n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 3, n) - begin_offset, (T) 0);

  fill_n_kernel<<<1, 1>>>(exec, d_data.begin() + begin_offset, std::min((size_t) 3, n) - begin_offset, (T) 0);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  begin_offset = std::min<size_t>(117, n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 367, n) - begin_offset, (T) 1);

  fill_n_kernel<<<1, 1>>>(exec, d_data.begin() + begin_offset, std::min((size_t) 367, n) - begin_offset, (T) 1);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  begin_offset = std::min<size_t>(8, n);

  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 259, n) - begin_offset, (T) 2);

  fill_n_kernel<<<1, 1>>>(exec, d_data.begin() + begin_offset, std::min((size_t) 259, n) - begin_offset, (T) 2);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  begin_offset = std::min<size_t>(3, n);

  thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, (T) 3);

  fill_n_kernel<<<1, 1>>>(exec, d_data.begin() + begin_offset, d_data.size() - begin_offset, (T) 3);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);

  thrust::fill_n(h_data.begin(), h_data.size(), (T) 4);

  fill_n_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.size(), (T) 4);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_data, d_data);
}

template <typename T>
void TestFillNDeviceSeq(size_t n)
{
  TestFillNDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceSeq);

template <typename T>
void TestFillNDeviceDevice(size_t n)
{
  TestFillNDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestFillNDeviceDevice);
#endif

void TestFillCudaStreams()
{
  thrust::device_vector<int> v{0, 1, 2, 3, 4};

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::fill(thrust::cuda::par.on(s), v.begin() + 1, v.begin() + 4, 7);
  cudaStreamSynchronize(s);

  thrust::device_vector<int> ref{0, 7, 7, 7, 4};
  ASSERT_EQUAL(v, ref);

  thrust::fill(thrust::cuda::par.on(s), v.begin() + 0, v.begin() + 3, 8);
  cudaStreamSynchronize(s);

  ref = {8, 8, 8, 7, 4};
  ASSERT_EQUAL(v, ref);

  thrust::fill(thrust::cuda::par.on(s), v.begin() + 2, v.end(), 9);
  cudaStreamSynchronize(s);

  ref = {8, 8, 9, 9, 9};
  ASSERT_EQUAL(v, ref);

  thrust::fill(thrust::cuda::par.on(s), v.begin(), v.end(), 1);
  cudaStreamSynchronize(s);

  ref = {1, 1, 1, 1, 1};
  ASSERT_EQUAL(v, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestFillCudaStreams);
