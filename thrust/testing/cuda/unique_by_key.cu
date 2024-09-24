#include <thrust/execution_policy.h>
#include <thrust/functional.h>
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

template <typename Vector>
void initialize_keys(Vector& keys)
{
  keys.resize(9);
  keys = {11, 11, 21, 20, 21, 21, 21, 37, 37};
}

template <typename Vector>
void initialize_values(Vector& values)
{
  values.resize(9);
  values = {0, 1, 2, 3, 4, 5, 6, 7, 8};
}

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void unique_by_key_kernel(
  ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  *result = thrust::unique_by_key(exec, keys_first, keys_last, values_first);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename BinaryPredicate, typename Iterator3>
__global__ void unique_by_key_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  BinaryPredicate pred,
  Iterator3 result)
{
  *result = thrust::unique_by_key(exec, keys_first, keys_last, values_first, pred);
}

template <typename ExecutionPolicy>
void TestUniqueByKeyDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector keys;
  Vector values;

  using iter_pair = thrust::pair<typename Vector::iterator, typename Vector::iterator>;
  thrust::device_vector<iter_pair> new_last_vec(1);
  iter_pair new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  unique_by_key_kernel<<<1, 1>>>(exec, keys.begin(), keys.end(), values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - keys.begin(), 5);
  keys.erase(new_last.first, keys.end());
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  ASSERT_EQUAL(new_last.second - values.begin(), 5);
  values.erase(new_last.second, values.end());
  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  unique_by_key_kernel<<<1, 1>>>(
    exec, keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - keys.begin(), 3);
  keys.erase(new_last.first, keys.end());
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  ASSERT_EQUAL(new_last.second - values.begin(), 3);
  values.erase(new_last.second, values.end());
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(values, values_ref);
}

void TestUniqueByKeyDeviceSeq()
{
  TestUniqueByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceSeq);

void TestUniqueByKeyDeviceDevice()
{
  TestUniqueByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceDevice);

void TestUniqueByKeyDeviceNoSync()
{
  TestUniqueByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueByKeyDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueByKeyCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector keys;
  Vector values;

  using iter_pair = thrust::pair<Vector::iterator, Vector::iterator>;
  iter_pair new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique_by_key(streampolicy, keys.begin(), keys.end(), values.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last.first - keys.begin(), 5);
  keys.erase(new_last.first, keys.end());
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  ASSERT_EQUAL(new_last.second - values.begin(), 5);
  values.erase(new_last.second, values.end());
  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::unique_by_key(streampolicy, keys.begin(), keys.end(), values.begin(), is_equal_div_10_unique<T>());

  ASSERT_EQUAL(new_last.first - keys.begin(), 3);
  keys.erase(new_last.first, keys.end());
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(keys, keys_ref);

  ASSERT_EQUAL(new_last.second - values.begin(), 3);
  values.erase(new_last.second, values.end());
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(values, values_ref);

  cudaStreamDestroy(s);
}

void TestUniqueByKeyCudaStreamsSync()
{
  TestUniqueByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueByKeyCudaStreamsSync);

void TestUniqueByKeyCudaStreamsNoSync()
{
  TestUniqueByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueByKeyCudaStreamsNoSync);

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Iterator5>
__global__ void unique_by_key_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  Iterator5 result)
{
  *result = thrust::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_result, values_result);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename BinaryPredicate,
          typename Iterator5>
__global__ void unique_by_key_copy_kernel(
  ExecutionPolicy exec,
  Iterator1 keys_first,
  Iterator1 keys_last,
  Iterator2 values_first,
  Iterator3 keys_result,
  Iterator4 values_result,
  BinaryPredicate pred,
  Iterator5 result)
{
  *result = thrust::unique_by_key_copy(exec, keys_first, keys_last, values_first, keys_result, values_result, pred);
}

template <typename ExecutionPolicy>
void TestUniqueCopyByKeyDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector keys;
  Vector values;

  using iter_pair = thrust::pair<typename Vector::iterator, typename Vector::iterator>;
  thrust::device_vector<iter_pair> new_last_vec(1);
  iter_pair new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  unique_by_key_copy_kernel<<<1, 1>>>(
    exec, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin(), new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  output_keys.erase(new_last.first, output_keys.end());
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  output_values.erase(new_last.second, output_values.end());
  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(output_values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  unique_by_key_copy_kernel<<<1, 1>>>(
    exec,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    is_equal_div_10_unique<T>(),
    new_last_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  new_last = new_last_vec[0];

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);

  output_keys.erase(new_last.first, output_keys.end());
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  output_values.erase(new_last.second, output_values.end());
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(output_values, values_ref);
}

void TestUniqueCopyByKeyDeviceSeq()
{
  TestUniqueCopyByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceSeq);

void TestUniqueCopyByKeyDeviceDevice()
{
  TestUniqueCopyByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceDevice);

void TestUniqueCopyByKeyDeviceNoSync()
{
  TestUniqueCopyByKeyDevice(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyDeviceNoSync);
#endif

template <typename ExecutionPolicy>
void TestUniqueCopyByKeyCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector keys;
  Vector values;

  using iter_pair = thrust::pair<Vector::iterator, Vector::iterator>;
  iter_pair new_last;

  // basic test
  initialize_keys(keys);
  initialize_values(values);

  Vector output_keys(keys.size());
  Vector output_values(values.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  auto streampolicy = policy.on(s);

  new_last = thrust::unique_by_key_copy(
    streampolicy, keys.begin(), keys.end(), values.begin(), output_keys.begin(), output_values.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 5);
  output_keys.erase(new_last.first, output_keys.end());
  Vector keys_ref{11, 21, 20, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  ASSERT_EQUAL(new_last.second - output_values.begin(), 5);
  output_values.erase(new_last.second, output_values.end());
  Vector values_ref{0, 2, 3, 4, 7};
  ASSERT_EQUAL(output_values, values_ref);

  // test BinaryPredicate
  initialize_keys(keys);
  initialize_values(values);

  new_last = thrust::unique_by_key_copy(
    streampolicy,
    keys.begin(),
    keys.end(),
    values.begin(),
    output_keys.begin(),
    output_values.begin(),
    is_equal_div_10_unique<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(new_last.first - output_keys.begin(), 3);
  output_keys.erase(new_last.first, output_keys.end());
  keys_ref = {11, 21, 37};
  ASSERT_EQUAL(output_keys, keys_ref);

  ASSERT_EQUAL(new_last.second - output_values.begin(), 3);
  output_values.erase(new_last.second, output_values.end());
  values_ref = {0, 2, 7};
  ASSERT_EQUAL(output_values, values_ref);

  cudaStreamDestroy(s);
}

void TestUniqueCopyByKeyCudaStreamsSync()
{
  TestUniqueCopyByKeyCudaStreams(thrust::cuda::par);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyCudaStreamsSync);

void TestUniqueCopyByKeyCudaStreamsNoSync()
{
  TestUniqueCopyByKeyCudaStreams(thrust::cuda::par_nosync);
}
DECLARE_UNITTEST(TestUniqueCopyByKeyCudaStreamsNoSync);
