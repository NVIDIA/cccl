#include <thrust/execution_policy.h>
#include <thrust/gather.h>

#include <algorithm>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void
gather_kernel(ExecutionPolicy exec, Iterator1 map_first, Iterator1 map_last, Iterator2 elements_first, Iterator3 result)
{
  thrust::gather(exec, map_first, map_last, elements_first, result);
}

template <typename T, typename ExecutionPolicy>
void TestGatherDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);

  // source vectors to gather from
  thrust::host_vector<T> h_source   = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;

  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % source_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  // gather destination
  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::gather(h_map.begin(), h_map.end(), h_source.begin(), h_output.begin());

  gather_kernel<<<1, 1>>>(exec, d_map.begin(), d_map.end(), d_source.begin(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_output, d_output);
}

template <typename T>
void TestGatherDeviceSeq(const size_t n)
{
  TestGatherDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherDeviceSeq);

template <typename T>
void TestGatherDeviceDevice(const size_t n)
{
  TestGatherDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherDeviceDevice);
#endif

void TestGatherCudaStreams()
{
  thrust::device_vector<int> map = {6, 2, 1, 7, 2}; // gather indices
  thrust::device_vector<int> src = {0, 1, 2, 3, 4, 5, 6, 7}; // source vector
  thrust::device_vector<int> dst = {0, 0, 0, 0, 0}; // destination vector

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::gather(thrust::cuda::par.on(s), map.begin(), map.end(), src.begin(), dst.begin());
  cudaStreamSynchronize(s);

  thrust::device_vector<int> ref = {6, 2, 1, 7, 2}; // destination vector

  ASSERT_EQUAL(dst, ref);
  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGatherCudaStreams);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Predicate>
__global__ void gather_if_kernel(
  ExecutionPolicy exec,
  Iterator1 map_first,
  Iterator1 map_last,
  Iterator2 stencil_first,
  Iterator3 elements_first,
  Iterator4 result,
  Predicate pred)
{
  thrust::gather_if(exec, map_first, map_last, stencil_first, elements_first, result, pred);
}

template <typename T>
struct is_even_gather_if
{
  _CCCL_HOST_DEVICE bool operator()(const T i) const
  {
    return (i % 2) == 0;
  }
};

template <typename T, typename ExecutionPolicy>
void TestGatherIfDevice(ExecutionPolicy exec, const size_t n)
{
  const size_t source_size = std::min((size_t) 10, 2 * n);

  // source vectors to gather from
  thrust::host_vector<T> h_source   = unittest::random_samples<T>(source_size);
  thrust::device_vector<T> d_source = h_source;

  // gather indices
  thrust::host_vector<unsigned int> h_map = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_map[i] = h_map[i] % source_size;
  }

  thrust::device_vector<unsigned int> d_map = h_map;

  // gather stencil
  thrust::host_vector<unsigned int> h_stencil = unittest::random_integers<unsigned int>(n);

  for (size_t i = 0; i < n; i++)
  {
    h_stencil[i] = h_stencil[i] % 2;
  }

  thrust::device_vector<unsigned int> d_stencil = h_stencil;

  // gather destination
  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::gather_if(
    h_map.begin(),
    h_map.end(),
    h_stencil.begin(),
    h_source.begin(),
    h_output.begin(),
    is_even_gather_if<unsigned int>());

  gather_if_kernel<<<1, 1>>>(
    exec,
    d_map.begin(),
    d_map.end(),
    d_stencil.begin(),
    d_source.begin(),
    d_output.begin(),
    is_even_gather_if<unsigned int>());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(h_output, d_output);
}

template <typename T>
void TestGatherIfDeviceSeq(const size_t n)
{
  TestGatherIfDevice<T>(thrust::seq, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfDeviceSeq);

template <typename T>
void TestGatherIfDeviceDevice(const size_t n)
{
  TestGatherIfDevice<T>(thrust::device, n);
}
DECLARE_VARIABLE_UNITTEST(TestGatherIfDeviceDevice);
#endif

void TestGatherIfCudaStreams()
{
  thrust::device_vector<int> flg{0, 1, 0, 1, 0}; // predicate array
  thrust::device_vector<int> map{6, 2, 1, 7, 2}; // gather indices
  thrust::device_vector<int> src{0, 1, 2, 3, 4, 5, 6, 7}; // source vector
  thrust::device_vector<int> dst(5, 0); // destination vector

  cudaStream_t s;
  cudaStreamCreate(&s);

  thrust::gather_if(thrust::cuda::par.on(s), map.begin(), map.end(), flg.begin(), src.begin(), dst.begin());
  cudaStreamSynchronize(s);

  thrust::device_vector<int> ref{0, 2, 0, 7, 0}; // destination vector

  ASSERT_EQUAL(dst, ref);
  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestGatherIfCudaStreams);
