#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#include <unittest/unittest.h>

template <typename T>
struct is_equal_div_10_unique
{
  _CCCL_HOST_DEVICE bool operator()(const T x, const T& y) const
  {
    return ((int) x / 10) == ((int) y / 10);
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void unique_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::unique(exec, first, last);
}

template <typename ExecutionPolicy, typename Iterator1, typename BinaryPredicate, typename Iterator2>
__global__ void
unique_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::unique(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestUniqueDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  unique_kernel<<<1, 1>>>(exec, data.begin(), data.end(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 7);
  data.erase(new_last, data.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37}; // should we consider calculating ref from std::algorithm if exists?
  ASSERT_EQUAL(data, ref);

  unique_kernel<<<1, 1>>>(exec, data.begin(), new_last, is_equal_div_10_unique<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);
}

void TestUniqueDeviceSeq()
{
  TestUniqueDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueDeviceSeq);

void TestUniqueDeviceDevice()
{
  TestUniqueDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueDeviceDevice);

void TestUniqueDeviceNoSync()
{
  TestUniqueDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique(streampolicy, data.begin(), data.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 7);
  data.erase(new_last, data.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(data, ref);

  new_last = thrust::unique(streampolicy, data.begin(), new_last, is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

void TestUniqueCudaStreamsSync()
{
  TestUniqueCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCudaStreamsSync);

void TestUniqueCudaStreamsNoSync()
{
  TestUniqueCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCudaStreamsNoSync);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void
unique_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Iterator3 result2)
{
  *result2 = thrust::unique_copy(exec, first, last, result1);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__ void unique_copy_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, BinaryPredicate pred, Iterator3 result2)
{
  *result2 = thrust::unique_copy(exec, first, last, result1, pred);
}

template <typename ExecutionPolicy>
void TestUniqueCopyDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  unique_copy_kernel<<<1, 1>>>(exec, data.begin(), data.end(), output.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - output.begin(), 7);
  output.erase(new_last, output.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(output, ref);

  unique_copy_kernel<<<1, 1>>>(
    exec, output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);
}

void TestUniqueCopyDeviceSeq()
{
  TestUniqueCopyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceSeq);

void TestUniqueCopyDeviceDevice()
{
  TestUniqueCopyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceDevice);

void TestUniqueCopyDeviceNoSync()
{
  TestUniqueCopyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCopyCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(10, -1);

  thrust::device_vector<Vector::iterator> new_last_vec(1);
  Vector::iterator new_last;

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique_copy(streampolicy, data.begin(), data.end(), output.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - output.begin(), 7);
  output.erase(new_last, output.end());
  Vector ref{11, 12, 20, 29, 21, 31, 37};
  ASSERT_EQUAL(output, ref);

  new_last = thrust::unique_copy(streampolicy, output.begin(), new_last, data.begin(), is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last - data.begin(), 3);
  data.erase(new_last, data.end());
  ref = {11, 20, 31};
  ASSERT_EQUAL(data, ref);

  cudaStreamDestroy(s);
}

void TestUniqueCopyCudaStreamsSync()
{
  TestUniqueCopyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCopyCudaStreamsSync);

void TestUniqueCopyCudaStreamsNoSync()
{
  TestUniqueCopyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyCudaStreamsNoSync);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2>
__global__ void unique_count_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result)
{
  *result = thrust::unique_count(exec, first, last);
}

template <typename ExecutionPolicy, typename Iterator1, typename BinaryPredicate, typename Iterator2>
__global__ void
unique_count_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, BinaryPredicate pred, Iterator2 result)
{
  *result = thrust::unique_count(exec, first, last, pred);
}

template <typename ExecutionPolicy>
void TestUniqueCountDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  Vector output(1, -1);

  unique_count_kernel<<<1, 1>>>(exec, data.begin(), data.end(), output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(output[0], 7);

  unique_count_kernel<<<1, 1>>>(exec, data.begin(), data.end(), is_equal_div_10_unique<T>(), output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ASSERT_EQUAL(output[0], 3);
}

void TestUniqueCountDeviceSeq()
{
  TestUniqueCountDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCountDeviceSeq);

void TestUniqueCountDeviceDevice()
{
  TestUniqueCountDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCountDeviceDevice);

void TestUniqueCountDeviceNoSync()
{
  TestUniqueCountDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCountDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCountCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{11, 11, 12, 20, 29, 21, 21, 31, 31, 37};

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  int result = thrust::unique_count(streampolicy, data.begin(), data.end());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(result, 7);

  result = thrust::unique_count(streampolicy, data.begin(), data.end(), is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(result, 3);

  cudaStreamDestroy(s);
}

void TestUniqueCountCudaStreamsSync()
{
  TestUniqueCountCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCountCudaStreamsSync);

void TestUniqueCountCudaStreamsNoSync()
{
  TestUniqueCountCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCountCudaStreamsNoSync);
