#include <thrust/execution_policy.h>
#include <thrust/scatter.h>

#include <algorithm>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void
scatter_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 map_first, Iterator3 result)
{
  thrust::scatter(exec, first, last, map_first, result);
}

template <typename ExecutionPolicy>
void TestScatterDevice(ExecutionPolicy exec)
{
  size_t n                 = 1000;
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<int> h_input(n, 1);
  thrust::device_vector<int> d_input(n, 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::host_vector<int> h_output(output_size, 0);
  thrust::device_vector<int> d_output(output_size, 0);

  thrust::scatter(h_input.begin(), h_input.end(), h_map.begin(), h_output.begin());

  scatter_kernel<<<1, 1>>>(exec, d_input.begin(), d_input.end(), d_map.begin(), d_output.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_output, d_output);
}

void TestScatterDeviceSeq()
{
  TestScatterDevice(thrust::seq);
}
DECLARE_UNITTEST(TestScatterDeviceSeq);

void TestScatterDeviceDevice()
{
  TestScatterDevice(thrust::device);
}
DECLARE_UNITTEST(TestScatterDeviceDevice);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Function>
__global__ void scatter_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 map_first,
  Iterator3 stencil_first,
  Iterator4 result,
  Function f)
{
  thrust::scatter_if(exec, first, last, map_first, stencil_first, result, f);
}

template <typename T>
struct is_even_scatter_if
{
  _CCCL_HOST_DEVICE bool operator()(const T i) const
  {
    return (i % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestScatterIfDevice(ExecutionPolicy exec)
{
  size_t n                 = 1000;
  const size_t output_size = std::min((size_t) 10, 2 * n);

  thrust::host_vector<int> h_input(n, 1);
  thrust::device_vector<int> d_input(n, 1);

  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % output_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  thrust::host_vector<int> h_output(output_size, 0);
  thrust::device_vector<int> d_output(output_size, 0);

  thrust::scatter_if(
    h_input.begin(), h_input.end(), h_map.begin(), h_map.begin(), h_output.begin(), is_even_scatter_if<unsigned int>());

  scatter_if_kernel<<<1, 1>>>(
    exec,
    d_input.begin(),
    d_input.end(),
    d_map.begin(),
    d_map.begin(),
    d_output.begin(),
    is_even_scatter_if<unsigned int>());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  ASSERT_EQUAL(h_output, d_output);
}

void TestScatterIfDeviceSeq()
{
  TestScatterIfDevice(thrust::seq);
}
DECLARE_UNITTEST(TestScatterIfDeviceSeq);

void TestScatterIfDeviceDevice()
{
  TestScatterIfDevice(thrust::device);
}
DECLARE_UNITTEST(TestScatterIfDeviceDevice);
#endif

void TestScatterCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector map{6, 3, 1, 7, 2}; // scatter indices
  Vector src{0, 1, 2, 3, 4}; // source vector
  Vector dst(8, 0); // destination vector

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::scatter(thrust::cuda::par.on(s), src.begin(), src.end(), map.begin(), dst.begin());

  cudaStreamSynchronize(s);

  Vector ref{0, 2, 4, 1, 0, 0, 0, 3};
  ASSERT_EQUAL(dst, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestScatterCudaStreams);

void TestScatterIfCudaStreams()
{
  using Vector = thrust::device_vector<int>;

  Vector flg{0, 1, 0, 1, 0}; // predicate array
  Vector map{6, 3, 1, 7, 2}; // scatter indices
  Vector src{0, 1, 2, 3, 4}; // source vector
  Vector dst(8); // destination vector

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::scatter_if(thrust::cuda::par.on(s), src.begin(), src.end(), map.begin(), flg.begin(), dst.begin());
  cudaStreamSynchronize(s);

  Vector ref{0, 0, 0, 1, 0, 0, 0, 3};
  ASSERT_EQUAL(dst, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestScatterIfCudaStreams);
