#include <thrust/execution_policy.h>
#include <thrust/extrema.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename Iterator2>
__global__ void max_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Iterator2 result)
{
  *result = thrust::max_element(exec, first, last);
}

template <typename ExecutionPolicy, typename Iterator, typename BinaryPredicate, typename Iterator2>
__global__ void
max_element_kernel(ExecutionPolicy exec, Iterator first, Iterator last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::max_element(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestMaxElementDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  using iter_type = typename thrust::device_vector<int>::iterator;

  thrust::device_vector<iter_type> d_result(1);

  typename thrust::host_vector<int>::iterator h_max = thrust::max_element(h_data.begin(), h_data.end());

  max_element_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(h_max - h_data.begin() == (iter_type) d_result[0] - d_data.begin());

  typename thrust::host_vector<int>::iterator h_min =
    thrust::max_element(h_data.begin(), h_data.end(), ::cuda::std::greater<int>());

  max_element_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), ::cuda::std::greater<int>(), d_result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(h_min - h_data.begin() == (iter_type) d_result[0] - d_data.begin());
}

void TestMaxElementDeviceSeq()
{
  TestMaxElementDevice(thrust::seq);
}
TEST_CASE("TestMaxElementDeviceSeq", "[max_element]")
{
  TestMaxElementDeviceSeq();
}

void TestMaxElementDeviceDevice()
{
  TestMaxElementDevice(thrust::device);
}
TEST_CASE("TestMaxElementDeviceDevice", "[max_element]")
{
  TestMaxElementDeviceDevice();
}

void TestMaxElementDeviceNoSync()
{
  TestMaxElementDevice(thrust::cuda::par_nosync);
}
TEST_CASE("TestMaxElementDeviceNoSync", "[max_element]")
{
  TestMaxElementDeviceNoSync();
}
#endif

template <typename ExecutionPolicy>
void TestMaxElementCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data(6);
  data[0] = 3;
  data[1] = 5;
  data[2] = 1;
  data[3] = 2;
  data[4] = 5;
  data[5] = 1;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  REQUIRE(*thrust::max_element(streampolicy, data.begin(), data.end()) == 5);
  REQUIRE(thrust::max_element(streampolicy, data.begin(), data.end()) - data.begin() == 1);

  REQUIRE(*thrust::max_element(streampolicy, data.begin(), data.end(), ::cuda::std::greater<T>()) == 1);
  REQUIRE(thrust::max_element(streampolicy, data.begin(), data.end(), ::cuda::std::greater<T>()) - data.begin() == 2);

  cudaStreamDestroy(s);
}

void TestMaxElementCudaStreamsSync()
{
  TestMaxElementCudaStreams(thrust::cuda::par);
}
TEST_CASE("TestMaxElementCudaStreamsSync", "[max_element]")
{
  TestMaxElementCudaStreamsSync();
}

void TestMaxElementCudaStreamsNoSync()
{
  TestMaxElementCudaStreams(thrust::cuda::par_nosync);
}
TEST_CASE("TestMaxElementCudaStreamsNoSync", "[max_element]")
{
  TestMaxElementCudaStreamsNoSync();
}

void TestMaxElementDevicePointer()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data(6);
  data[0] = 3;
  data[1] = 5;
  data[2] = 1;
  data[3] = 2;
  data[4] = 5;
  data[5] = 1;

  T* raw_ptr     = thrust::raw_pointer_cast(data.data());
  const size_t n = data.size();
  REQUIRE(thrust::max_element(thrust::device, raw_ptr, raw_ptr + n) - raw_ptr == 1);
  REQUIRE(thrust::max_element(thrust::device, raw_ptr, raw_ptr + n, ::cuda::std::greater<T>()) - raw_ptr == 2);
}
TEST_CASE("TestMaxElementDevicePointer", "[max_element]")
{
  TestMaxElementDevicePointer();
}
