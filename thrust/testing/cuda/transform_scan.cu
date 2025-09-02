#include <thrust/execution_policy.h>
#include <thrust/transform_scan.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Function1,
          typename Function2,
          typename Iterator3>
__global__ void transform_inclusive_scan_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 result1,
  Function1 f1,
  Function2 f2,
  Iterator3 result2)
{
  *result2 = thrust::transform_inclusive_scan(exec, first, last, result1, f1, f2);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Function1,
          typename T,
          typename Function2,
          typename Iterator3>
__global__ void transform_inclusive_scan_init_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 result1,
  Function1 f1,
  T init,
  Function2 f2,
  Iterator3 result2)
{
  *result2 = thrust::transform_inclusive_scan(exec, first, last, result1, f1, init, f2);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Function1,
          typename T,
          typename Function2,
          typename Iterator3>
__global__ void transform_exclusive_scan_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 result,
  Function1 f1,
  T init,
  Function2 f2,
  Iterator3 result2)
{
  *result2 = thrust::transform_exclusive_scan(exec, first, last, result, f1, init, f2);
}

template <typename ExecutionPolicy>
void TestTransformScanDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, 3, -2, 4, -5};
  Vector ref{-1, -4, -2, -6, -1};
  Vector output(5);

  Vector input_copy(input);

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  // inclusive scan
  transform_inclusive_scan_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(ref, output);

  // inclusive scan with nonzero init
  transform_inclusive_scan_init_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref  = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(ref, output);

  // exclusive scan with 0 init
  transform_exclusive_scan_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    0,
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  ref = {0, -1, -4, -2, -6};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(ref, output);

  // exclusive scan with nonzero init
  transform_exclusive_scan_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref  = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(ref, output);

  // inplace inclusive scan
  input = input_copy;
  transform_inclusive_scan_kernel<<<1, 1>>>(
    exec, input.begin(), input.end(), input.begin(), ::cuda::std::negate<T>(), ::cuda::std::plus<T>(), iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref  = {-1, -4, -2, -6, -1};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(ref, input);

  // inplace inclusive scan with init
  input = input_copy;
  transform_inclusive_scan_init_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    input.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref  = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(ref, input);

  // inplace exclusive scan with init
  input = input_copy;
  transform_exclusive_scan_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    input.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>(),
    iter_vec.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    ASSERT_EQUAL(cudaSuccess, err);
  }

  iter = iter_vec[0];
  ref  = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(ref, input);
}

void TestTransformScanDeviceSeq()
{
  TestTransformScanDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformScanDeviceSeq);

void TestTransformScanDeviceDevice()
{
  TestTransformScanDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformScanDeviceDevice);
#endif

void TestTransformScanCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input{1, 3, -2, 4, -5};
  Vector result{-1, -4, -2, -6, -1};
  Vector output(5);

  Vector input_copy(input);

  cudaStream_t s;
  cudaStreamCreate(&s);

  // inclusive scan
  iter = thrust::transform_inclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inclusive scan with nonzero init
  iter = thrust::transform_inclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with 0 init
  iter = thrust::transform_exclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    0,
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {0, -1, -4, -2, -6};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // exclusive scan with nonzero init
  iter = thrust::transform_exclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(input, input_copy);
  ASSERT_EQUAL(output, result);

  // inplace inclusive scan
  input = input_copy;
  iter  = thrust::transform_inclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    input.begin(),
    ::cuda::std::negate<T>(),
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {-1, -4, -2, -6, -1};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace inclusive scan with init
  input = input_copy;
  iter  = thrust::transform_inclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    input.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {2, -1, 1, -3, 2};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  // inplace exclusive scan with init
  input = input_copy;
  iter  = thrust::transform_exclusive_scan(
    thrust::cuda::par.on(s),
    input.begin(),
    input.end(),
    input.begin(),
    ::cuda::std::negate<T>(),
    3,
    ::cuda::std::plus<T>());
  cudaStreamSynchronize(s);

  result = {3, 2, -1, 1, -3};
  ASSERT_EQUAL(std::size_t(iter - input.begin()), input.size());
  ASSERT_EQUAL(input, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformScanCudaStreams);

void TestTransformScanConstAccumulator()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input{1, 3, -2, 4, -5};
  Vector reference(5);
  Vector output(5);

  thrust::transform_inclusive_scan(
    input.begin(), input.end(), output.begin(), ::cuda::std::identity{}, ::cuda::std::plus<T>());
  thrust::inclusive_scan(input.begin(), input.end(), reference.begin(), ::cuda::std::plus<T>());

  ASSERT_EQUAL(output, reference);
}
DECLARE_UNITTEST(TestTransformScanConstAccumulator);
