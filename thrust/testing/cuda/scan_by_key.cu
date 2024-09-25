#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scan.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void inclusive_scan_by_key_kernel(
  ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::inclusive_scan_by_key(exec, keys_first, keys_last, values_first, result);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3>
__global__ void exclusive_scan_by_key_kernel(
  ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result)
{
  thrust::exclusive_scan_by_key(exec, keys_first, keys_last, values_first, result);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Iterator3, typename T>
__global__ void exclusive_scan_by_key_kernel(
  ExecutionPolicy exec, Iterator1 keys_first, Iterator1 keys_last, Iterator2 values_first, Iterator3 result, T init)
{
  thrust::exclusive_scan_by_key(exec, keys_first, keys_last, values_first, result, init);
}

template <typename ExecutionPolicy>
void TestScanByKeyDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  thrust::host_vector<int> h_keys(n);
  for (size_t i = 0, k = 0; i < n; i++)
  {
    h_keys[i] = static_cast<int>(k);
    if (rand() % 10 == 0)
    {
      k++;
    }
  }
  thrust::device_vector<int> d_keys = h_keys;

  thrust::host_vector<int> h_vals = unittest::random_integers<int>(n);
  for (size_t i = 0; i < n; i++)
  {
    h_vals[i] = i % 10;
  }
  thrust::device_vector<int> d_vals = h_vals;

  thrust::host_vector<int> h_output(n);
  thrust::device_vector<int> d_output(n);

  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  inclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_output, h_output);

  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  exclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_output, h_output);

  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), 11);
  exclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_output.begin(), 11);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_output, h_output);

  // in-place scans: in/out values aliasing
  h_output = h_vals;
  d_output = d_vals;
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin());
  inclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_output, h_output);

  h_output = h_vals;
  d_output = d_vals;
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_output.begin(), h_output.begin(), 11);
  exclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_output.begin(), d_output.begin(), 11);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_output, h_output);

  // in-place scans: keys/values aliasing
  thrust::inclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin());
  inclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_keys, h_output);

  d_keys = h_keys;
  thrust::exclusive_scan_by_key(h_keys.begin(), h_keys.end(), h_vals.begin(), h_output.begin(), 11);
  exclusive_scan_by_key_kernel<<<1, 1>>>(exec, d_keys.begin(), d_keys.end(), d_vals.begin(), d_keys.begin(), 11);
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }
  ASSERT_EQUAL(d_keys, h_output);
}

void TestScanByKeyDeviceSeq()
{
  TestScanByKeyDevice(thrust::seq);
}
DECLARE_UNITTEST(TestScanByKeyDeviceSeq);

void TestScanByKeyDeviceDevice()
{
  TestScanByKeyDevice(thrust::device);
}
DECLARE_UNITTEST(TestScanByKeyDeviceDevice);
#endif

void TestInclusiveScanByKeyCudaStreams()
{
  using Vector   = thrust::device_vector<int>;
  using T        = Vector::value_type;
  using Iterator = Vector::iterator;

  Vector keys{0, 1, 1, 1, 2, 3, 3};
  Vector vals{1, 2, 3, 4, 5, 6, 7};

  Vector output(7, 0);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator iter =
    thrust::inclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(iter, output.end());

  Vector ref{1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQUAL(output, ref);

  thrust::inclusive_scan_by_key(
    thrust::cuda::par.on(s),
    keys.begin(),
    keys.end(),
    vals.begin(),
    output.begin(),
    thrust::equal_to<T>(),
    thrust::multiplies<T>());
  cudaStreamSynchronize(s);

  ref = {1, 2, 6, 24, 5, 6, 42};
  ASSERT_EQUAL(output, ref);

  thrust::inclusive_scan_by_key(
    thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), thrust::equal_to<T>());
  cudaStreamSynchronize(s);

  ref = {1, 2, 5, 9, 5, 6, 13};
  ASSERT_EQUAL(output, ref);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestInclusiveScanByKeyCudaStreams);

void TestExclusiveScanByKeyCudaStreams()
{
  using Vector   = thrust::device_vector<int>;
  using T        = Vector::value_type;
  using Iterator = Vector::iterator;

  Vector keys{0, 1, 1, 1, 2, 3, 3};
  Vector vals{1, 2, 3, 4, 5, 6, 7};

  Vector output(7, 0);

  cudaStream_t s;
  cudaStreamCreate(&s);

  Iterator iter =
    thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL_QUIET(iter, output.end());

  Vector ref{0, 0, 2, 5, 0, 0, 6};
  ASSERT_EQUAL(output, ref);

  thrust::exclusive_scan_by_key(thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), T(10));
  cudaStreamSynchronize(s);

  ref = {10, 10, 12, 15, 10, 10, 16};
  ASSERT_EQUAL(output, ref);

  thrust::exclusive_scan_by_key(
    thrust::cuda::par.on(s),
    keys.begin(),
    keys.end(),
    vals.begin(),
    output.begin(),
    T(10),
    thrust::equal_to<T>(),
    thrust::multiplies<T>());
  cudaStreamSynchronize(s);

  ref = {10, 10, 20, 60, 10, 10, 60};
  ASSERT_EQUAL(output, ref);

  thrust::exclusive_scan_by_key(
    thrust::cuda::par.on(s), keys.begin(), keys.end(), vals.begin(), output.begin(), T(10), thrust::equal_to<T>());
  cudaStreamSynchronize(s);

  ref = {10, 10, 12, 15, 10, 10, 16};
  ASSERT_EQUAL(output, ref);
}
DECLARE_UNITTEST(TestExclusiveScanByKeyCudaStreams);
