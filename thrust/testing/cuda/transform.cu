#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <cuda/iterator>

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

  transform_kernel<<<1, 1>>>(
    exec, input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  iter = iter_vec[0];

  REQUIRE(std::size_t(iter - output.begin()) == input.size());
  REQUIRE(output == result);
}

TEST_CASE("TestTransformUnaryDeviceSeq", "[transform]")
{
  TestTransformUnaryDevice(thrust::seq);
}

TEST_CASE("TestTransformUnaryDeviceDevice", "[transform]")
{
  TestTransformUnaryDevice(thrust::device);
}

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
    exec,
    input.begin(),
    input.end(),
    output.begin(),
    ::cuda::std::negate<T>(),
    ::cuda::std::identity{},
    iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  iter = iter_vec[0];

  REQUIRE(std::size_t(iter - output.begin()) == input.size());
  REQUIRE(output == result);
}

TEST_CASE("TestTransformIfUnaryNoStencilDeviceSeq", "[transform]")
{
  TestTransformIfUnaryNoStencilDevice(thrust::seq);
}

TEST_CASE("TestTransformIfUnaryNoStencilDeviceDevice", "[transform]")
{
  TestTransformIfUnaryNoStencilDevice(thrust::device);
}

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
    ::cuda::std::negate<T>(),
    ::cuda::std::identity{},
    iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  iter = iter_vec[0];

  REQUIRE(std::size_t(iter - output.begin()) == input.size());
  REQUIRE(output == result);
}

TEST_CASE("TestTransformIfUnaryDeviceSeq", "[transform]")
{
  TestTransformIfUnaryDevice(thrust::seq);
}

TEST_CASE("TestTransformIfUnaryDeviceDevice", "[transform]")
{
  TestTransformIfUnaryDevice(thrust::device);
}

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
    exec, input1.begin(), input1.end(), input2.begin(), output.begin(), ::cuda::std::minus<T>(), iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  iter = iter_vec[0];

  REQUIRE(std::size_t(iter - output.begin()) == input1.size());
  REQUIRE(output == result);
}

TEST_CASE("TestTransformBinaryDeviceSeq", "[transform]")
{
  TestTransformBinaryDevice(thrust::seq);
}

TEST_CASE("TestTransformBinaryDeviceDevice", "[transform]")
{
  TestTransformBinaryDevice(thrust::device);
}

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

  ::cuda::std::identity identity;

  thrust::device_vector<typename Vector::iterator> iter_vec(1);

  transform_if_kernel<<<1, 1>>>(
    exec,
    input1.begin(),
    input1.end(),
    input2.begin(),
    stencil.begin(),
    output.begin(),
    ::cuda::std::minus<T>(),
    ::cuda::std::not_fn(identity),
    iter_vec.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  iter = iter_vec[0];

  REQUIRE(std::size_t(iter - output.begin()) == input1.size());
  REQUIRE(output == result);
}

TEST_CASE("TestTransformIfBinaryDeviceSeq", "[transform]")
{
  TestTransformIfBinaryDevice(thrust::seq);
}

TEST_CASE("TestTransformIfBinaryDeviceDevice", "[transform]")
{
  TestTransformIfBinaryDevice(thrust::device);
}
#endif

TEST_CASE("TestTransformUnaryCudaStreams", "[transform]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input{1, -2, 3};
  Vector output(3);
  const Vector result{-1, 2, -3};

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter =
    thrust::transform(thrust::cuda::par.on(s), input.begin(), input.end(), output.begin(), ::cuda::std::negate<T>());
  cudaStreamSynchronize(s);

  REQUIRE(std::size_t(iter - output.begin()) == input.size());
  REQUIRE(output == result);

  cudaStreamDestroy(s);
}

TEST_CASE("TestTransformBinaryCudaStreams", "[transform]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector::iterator iter;

  Vector input1{1, -2, 3};
  Vector input2{-4, 5, 6};
  Vector output(3);
  const Vector result{5, -7, -3};

  cudaStream_t s;
  cudaStreamCreate(&s);

  iter = thrust::transform(
    thrust::cuda::par.on(s), input1.begin(), input1.end(), input2.begin(), output.begin(), ::cuda::std::minus<T>());
  cudaStreamSynchronize(s);

  REQUIRE(std::size_t(iter - output.begin()) == input1.size());
  REQUIRE(output == result);

  cudaStreamDestroy(s);
}

struct sum_five
{
  _CCCL_HOST_DEVICE auto operator()(std::int8_t a, std::int16_t b, std::int32_t c, std::int64_t d, float e) const
    -> double
  {
    return static_cast<double>(a) + static_cast<double>(b) + static_cast<double>(c) + static_cast<double>(d)
         + static_cast<double>(e);
  }
};

// we specialize zip_function for sum_five, but do nothing in the call operator so the test below would fail if the
// zip_function is actually called (and not unwrapped)
THRUST_NAMESPACE_BEGIN
template <>
class zip_function<sum_five>
{
public:
  _CCCL_HOST_DEVICE zip_function(sum_five func)
      : func(func)
  {}

  _CCCL_HOST_DEVICE sum_five& underlying_function() const
  {
    return func;
  }

  template <typename Tuple>
  _CCCL_HOST_DEVICE double operator()(Tuple&& t) const
  {
    // not calling func, so we would get a wrong result if we were called
    return 0;
  }

private:
  mutable sum_five func;
};
THRUST_NAMESPACE_END

// test that the cuda_cub backend of Thrust unwraps zip_iterators/zip_functions into their input streams
TEST_CASE("TestTransformThrustZipIteratorUnwrapping", "[transform]")
{
  constexpr int num_items = 100;
  thrust::device_vector<std::int8_t> a(num_items, 1);
  thrust::device_vector<std::int16_t> b(num_items, 2);
  thrust::device_vector<std::int32_t> c(num_items, 3);
  thrust::device_vector<std::int64_t> d(num_items, 4);
  thrust::device_vector<float> e(num_items, 5);

  thrust::device_vector<double> result(num_items);
  // SECTION("once") // TODO(bgruber): enable sections when we migrate to Catch2
  {
    const auto z = thrust::make_zip_iterator(a.begin(), b.begin(), c.begin(), d.begin(), e.begin());
    thrust::transform(z, z + num_items, result.begin(), thrust::make_zip_function(sum_five{}));

    // compute reference and verify
    const thrust::device_vector<double> reference(num_items, 1 + 2 + 3 + 4 + 5);
    REQUIRE(reference == result);
  }
  // SECTION("trice")
  {
    const auto z = thrust::make_zip_iterator(
      thrust::make_zip_iterator(thrust::make_zip_iterator(a.begin(), b.begin(), c.begin(), d.begin(), e.begin())));
    thrust::transform(z,
                      z + num_items,
                      result.begin(),
                      thrust::make_zip_function(thrust::make_zip_function(thrust::make_zip_function(sum_five{}))));

    // compute reference and verify
    const thrust::device_vector<double> reference(num_items, 1 + 2 + 3 + 4 + 5);
    REQUIRE(reference == result);
  }
}

// we specialize zip_function for sum_five, but do nothing in the call operator so the test below would fail if the
// zip_function is actually called (and not unwrapped)
_CCCL_BEGIN_NAMESPACE_CUDA
template <>
class zip_function<sum_five>
{
private:
  sum_five __fun_;

public:
  zip_function() = default;

  _CCCL_HOST_DEVICE zip_function(sum_five&& func) noexcept
      : __fun_(::cuda::std::move(func))
  {}

  template <typename Tuple>
  _CCCL_HOST_DEVICE decltype(auto) operator()(Tuple&& tuple) const noexcept
  {
    // not calling func, just return a default ctored element, so we would get a wrong result if we were called
    return decltype(::cuda::std::apply(__fun_, ::cuda::std::forward<Tuple>(tuple))){};
  }

  // NOTE: this must be named __fun() to match the real cuda::zip_function API, which
  // cub::device_transform.cuh calls directly (without any existence check) when unwrapping.
  _CCCL_HOST_DEVICE sum_five& __fun() noexcept
  {
    return __fun_;
  }

  _CCCL_HOST_DEVICE const sum_five& __fun() const noexcept
  {
    return __fun_;
  }
};
_CCCL_END_NAMESPACE_CUDA

// test that the cuda_cub backend of Thrust unwraps zip_iterators/zip_functions into their input streams
TEST_CASE("TestTransformCudaZipIteratorUnwrapping", "[transform]")
{
  constexpr int num_items = 100;
  thrust::device_vector<std::int8_t> a(num_items, 1);
  thrust::device_vector<std::int16_t> b(num_items, 2);
  thrust::device_vector<std::int32_t> c(num_items, 3);
  thrust::device_vector<std::int64_t> d(num_items, 4);
  thrust::device_vector<float> e(num_items, 5);

  thrust::device_vector<double> result(num_items);
  // SECTION("once") // TODO(bgruber): enable sections when we migrate to Catch2
  {
    const auto z = cuda::make_zip_iterator(a.begin(), b.begin(), c.begin(), d.begin(), e.begin());
    thrust::transform(z, z + num_items, result.begin(), cuda::zip_function(sum_five{}));

    // compute reference and verify
    const thrust::device_vector<double> reference(num_items, 1 + 2 + 3 + 4 + 5);
    REQUIRE(reference == result);
  }
  // SECTION("trice")
  {
    const auto z = cuda::make_zip_iterator(
      cuda::make_zip_iterator(cuda::make_zip_iterator(a.begin(), b.begin(), c.begin(), d.begin(), e.begin())));
    thrust::transform(
      z, z + num_items, result.begin(), cuda::zip_function<cuda::zip_function<cuda::zip_function<sum_five>>>{});

    // compute reference and verify
    const thrust::device_vector<double> reference(num_items, 1 + 2 + 3 + 4 + 5);
    REQUIRE(reference == result);
  }
}
