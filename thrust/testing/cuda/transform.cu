#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Function, typename Iterator3>
__global__ void transform_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Function f, Iterator3 result2)
{
  *result2 = thrust::transform(exec, first, last, result1, f);
}

template <typename ExecutionPolicy>
void TestTransformUnaryDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector output(3);
  Vector result{-1, 2, -3};

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_kernel<<<1, 1>>>(exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformUnaryDeviceSeq()
{
  TestTransformUnaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformUnaryDeviceSeq);

void TestTransformUnaryDeviceDevice()
{
  TestTransformUnaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformUnaryDeviceDevice);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Function,
          typename Predicate,
          typename Iterator3>
__global__ void transform_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 result1,
  Function f,
  Predicate pred,
  Iterator3 result2)
{
  *result2 = thrust::transform_if(exec, first, last, result1, f, pred);
}

template <typename ExecutionPolicy>
void TestTransformIfUnaryNoStencilDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{0, -2, 0};
  Vector output{-1, -2, -3};
  Vector result{-1, 2, -3};

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_if_kernel<<<1, 1>>>(
    exec, input.begin(), input.end(), output.begin(), thrust::negate<T>(), thrust::identity<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfUnaryNoStencilDeviceSeq()
{
  TestTransformIfUnaryNoStencilDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfUnaryNoStencilDeviceSeq);

void TestTransformIfUnaryNoStencilDeviceDevice()
{
  TestTransformIfUnaryNoStencilDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfUnaryNoStencilDeviceDevice);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Function,
          typename Predicate,
          typename Iterator4>
__global__ void transform_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 stencil_first,
  Iterator3 result1,
  Function f,
  Predicate pred,
  Iterator4 result2)
{
  *result2 = thrust::transform_if(exec, first, last, stencil_first, result1, f, pred);
}

template <typename ExecutionPolicy>
void TestTransformIfUnaryDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector stencil{1, 0, 1};
  Vector output{1, 2, 3};
  Vector result{-1, 2, -3};

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_if_kernel<<<1, 1>>>(
    exec,
    input.begin(),
    input.end(),
    stencil.begin(),
    output.begin(),
    thrust::negate<T>(),
    thrust::identity<T>(),
    iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfUnaryDeviceSeq()
{
  TestTransformIfUnaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfUnaryDeviceSeq);

void TestTransformIfUnaryDeviceDevice()
{
  TestTransformIfUnaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfUnaryDeviceDevice);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Function,
          typename Iterator4>
__global__ void transform_kernel(
  ExecutionPolicy exec,
  Iterator1 first1,
  Iterator1 last1,
  Iterator2 first2,
  Iterator3 result1,
  Function f,
  Iterator4 result2)
{
  *result2 = thrust::transform(exec, first1, last1, first2, result1, f);
}

template <typename ExecutionPolicy>
void TestTransformBinaryDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector output(3);
  Vector result{5, -7, -3};

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_kernel<<<1, 1>>>(
    exec, input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformBinaryDeviceSeq()
{
  TestTransformBinaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformBinaryDeviceSeq);

void TestTransformBinaryDeviceDevice()
{
  TestTransformBinaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformBinaryDeviceDevice);

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Iterator4,
          typename Function,
          typename Predicate,
          typename Iterator5>
__global__ void transform_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first1,
  Iterator1 last1,
  Iterator2 first2,
  Iterator3 stencil_first,
  Iterator4 result1,
  Function f,
  Predicate pred,
  Iterator5 result2)
{
  *result2 = thrust::transform_if(exec, first1, last1, first2, stencil_first, result1, f, pred);
}

template <typename ExecutionPolicy>
void TestTransformIfBinaryDevice(ExecutionPolicy exec)
{
  using Vector = thrust::device_vector<int>;
  using T      = typename Vector::value_type;

  typename Vector::iterator iter;

  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector stencil{0, 1, 0};
  Vector output{1, 2, 3};
  Vector result{5, 2, -3};

  thrust::identity<T> identity;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_if_kernel<<<1, 1>>>(
    exec,
    input1.begin(),
    input1.end(),
    input2.begin(),
    stencil.begin(),
    output.begin(),
    thrust::minus<T>(),
    thrust::not_fn(identity),
    iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  ASSERT_EQUAL(cudaSuccess, err);

  iter = iter_vec[0];

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);
}

void TestTransformIfBinaryDeviceSeq()
{
  TestTransformIfBinaryDevice(thrust::seq);
}
DECLARE_UNITTEST(TestTransformIfBinaryDeviceSeq);

void TestTransformIfBinaryDeviceDevice()
{
  TestTransformIfBinaryDevice(thrust::device);
}
DECLARE_UNITTEST(TestTransformIfBinaryDeviceDevice);
#endif

void TestTransformUnaryCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector output(3);
  Vector result{-1, 2, -3};

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter = thrust::transform(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), thrust::negate<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input.size());
  ASSERT_EQUAL(output, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformUnaryCudaStreams);

void TestTransformBinaryCudaStreams()
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector output(3);
  Vector result{5, -7, -3};

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter = thrust::transform(
    thrust::cuda::par.on(s), input1.begin(), input1.end(), input2.begin(), output.begin(), thrust::minus<T>());
  cudaStreamSynchronize(s);

  ASSERT_EQUAL(std::size_t(iter - output.begin()), input1.size());
  ASSERT_EQUAL(output, result);

  cudaStreamDestroy(s);
}
DECLARE_UNITTEST(TestTransformBinaryCudaStreams);
