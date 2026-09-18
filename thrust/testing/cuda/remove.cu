#include <thrust/execution_policy.h>
#include <thrust/remove.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename T, typename Iterator2>
__global__ void remove_kernel(ExecutionPolicy exec, Iterator first, Iterator last, T val, Iterator2 result)
{
  *result = thrust::remove(exec, first, last, val);
}

template <typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__ void remove_if_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::remove_if(exec, first, last, pred);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void remove_if_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 stencil_first, Predicate pred, Iterator3 result)
{
  *result = thrust::remove_if(exec, first, last, stencil_first, pred);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename T, typename Iterator3>
__global__ void
remove_copy_kernel(ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, T val, Iterator3 result2)
{
  *result2 = thrust::remove_copy(exec, first, last, result1, val);
}

template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void remove_copy_if_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result, Predicate pred, Iterator3 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, result, pred);
}

template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Predicate,
          typename Iterator4>
__global__ void remove_copy_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 stencil_first,
  Iterator3 result,
  Predicate pred,
  Iterator4 result_end)
{
  *result_end = thrust::remove_copy_if(exec, first, last, stencil_first, result, pred);
}
#endif

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return (static_cast<unsigned int>(x) & 1) == 0;
  }
};

template <typename T>
struct is_true
{
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return x ? true : false;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy>
void TestRemoveDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_result(1);

  size_t h_size = thrust::remove(h_data.begin(), h_data.end(), 0) - h_data.begin();

  remove_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), 0, d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_result[0] - d_data.begin();

  REQUIRE(h_size == d_size);

  h_data.resize(h_size);
  d_data.resize(d_size);

  REQUIRE(h_data == d_data);
}

TEST_CASE("TestRemoveDeviceSeq", "[remove]")
{
  TestRemoveDevice(thrust::seq);
}

TEST_CASE("TestRemoveDeviceDevice", "[remove]")
{
  TestRemoveDevice(thrust::device);
}

template <typename ExecutionPolicy>
void TestRemoveIfDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_result(1);

  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), is_true<int>()) - h_data.begin();

  remove_if_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), is_true<int>(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_result[0] - d_data.begin();

  REQUIRE(h_size == d_size);

  h_data.resize(h_size);
  d_data.resize(d_size);

  REQUIRE(h_data == d_data);
}

TEST_CASE("TestRemoveIfDeviceSeq", "[remove]")
{
  TestRemoveIfDevice(thrust::seq);
}

TEST_CASE("TestRemoveIfDeviceDevice", "[remove]")
{
  TestRemoveIfDevice(thrust::device);
}

template <typename ExecutionPolicy>
void TestRemoveIfStencilDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_result(1);

  thrust::host_vector<bool> h_stencil   = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;

  size_t h_size = thrust::remove_if(h_data.begin(), h_data.end(), h_stencil.begin(), is_true<int>()) - h_data.begin();

  remove_if_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_stencil.begin(), is_true<int>(), d_result.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_result[0] - d_data.begin();

  REQUIRE(h_size == d_size);

  h_data.resize(h_size);
  d_data.resize(d_size);

  REQUIRE(h_data == d_data);
}

TEST_CASE("TestRemoveIfStencilDeviceSeq", "[remove]")
{
  TestRemoveIfStencilDevice(thrust::seq);
}

TEST_CASE("TestRemoveIfStencilDeviceDevice", "[remove]")
{
  TestRemoveIfStencilDevice(thrust::device);
}

template <typename ExecutionPolicy>
void TestRemoveCopyDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  thrust::host_vector<int> h_result(n);
  thrust::device_vector<int> d_result(n);

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_new_end(1);

  size_t h_size = thrust::remove_copy(h_data.begin(), h_data.end(), h_result.begin(), 0) - h_result.begin();

  remove_copy_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), 0, d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_new_end[0] - d_result.begin();

  REQUIRE(h_size == d_size);

  h_result.resize(h_size);
  d_result.resize(d_size);

  REQUIRE(h_result == d_result);
}

TEST_CASE("TestRemoveCopyDeviceSeq", "[remove]")
{
  TestRemoveCopyDevice(thrust::seq);
}

TEST_CASE("TestRemoveCopyDeviceDevice", "[remove]")
{
  TestRemoveCopyDevice(thrust::device);
}

template <typename ExecutionPolicy>
void TestRemoveCopyIfDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  thrust::host_vector<int> h_result(n);
  thrust::device_vector<int> d_result(n);

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_new_end(1);

  size_t h_size =
    thrust::remove_copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_true<int>()) - h_result.begin();

  remove_copy_if_kernel<<<1, 1>>>(
    exec, d_data.begin(), d_data.end(), d_result.begin(), is_true<int>(), d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_new_end[0] - d_result.begin();

  REQUIRE(h_size == d_size);

  h_result.resize(h_size);
  d_result.resize(d_size);

  REQUIRE(h_result == d_result);
}

TEST_CASE("TestRemoveCopyIfDeviceSeq", "[remove]")
{
  TestRemoveCopyIfDevice(thrust::seq);
}

TEST_CASE("TestRemoveCopyIfDeviceDevice", "[remove]")
{
  TestRemoveCopyIfDevice(thrust::device);
}

template <typename ExecutionPolicy>
void TestRemoveCopyIfStencilDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_samples<int>(n);
  thrust::device_vector<int> d_data = h_data;

  thrust::host_vector<int> h_result(n);
  thrust::device_vector<int> d_result(n);

  using iterator = typename thrust::device_vector<int>::iterator;
  thrust::device_vector<iterator> d_new_end(1);

  thrust::host_vector<bool> h_stencil   = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_stencil = h_stencil;

  size_t h_size =
    thrust::remove_copy_if(h_data.begin(), h_data.end(), h_stencil.begin(), h_result.begin(), is_true<int>())
    - h_result.begin();

  remove_copy_if_kernel<<<1, 1>>>(
    exec, d_data.begin(), d_data.end(), d_stencil.begin(), d_result.begin(), is_true<int>(), d_new_end.begin());
  cudaError_t const err = cudaDeviceSynchronize();
  REQUIRE(cudaSuccess == err);

  size_t d_size = (iterator) d_new_end[0] - d_result.begin();

  REQUIRE(h_size == d_size);

  h_result.resize(h_size);
  d_result.resize(d_size);

  REQUIRE(h_result == d_result);
}

TEST_CASE("TestRemoveCopyIfStencilDeviceSeq", "[remove]")
{
  TestRemoveCopyIfStencilDevice(thrust::seq);
}

TEST_CASE("TestRemoveCopyIfStencilDeviceDevice", "[remove]")
{
  TestRemoveCopyIfStencilDevice(thrust::device);
}
#endif

TEST_CASE("TestRemoveCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end = thrust::remove(thrust::cuda::par.on(s), data.begin(), data.end(), (T) 2);

  REQUIRE(end - data.begin() == 3);
  data.erase(end, data.end());

  const Vector ref{1, 1, 3};
  REQUIRE(data == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestRemoveCopyCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end =
    thrust::remove_copy(thrust::cuda::par.on(s), data.begin(), data.end(), result.begin(), (T) 2);

  REQUIRE(end - result.begin() == 3);
  result.erase(end, result.end());

  const Vector ref{1, 1, 3};
  REQUIRE(result == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestRemoveIfCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end = thrust::remove_if(thrust::cuda::par.on(s), data.begin(), data.end(), is_even<T>());

  REQUIRE(end - data.begin() == 3);
  data.erase(end, data.end());

  const Vector ref{1, 1, 3};
  REQUIRE(data == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestRemoveIfStencilCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector stencil{0, 1, 0, 0, 1};

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end =
    thrust::remove_if(thrust::cuda::par.on(s), data.begin(), data.end(), stencil.begin(), ::cuda::std::identity{});

  REQUIRE(end - data.begin() == 3);
  data.erase(end, data.end());

  const Vector ref{1, 1, 3};
  REQUIRE(data == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestRemoveCopyIfCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end =
    thrust::remove_copy_if(thrust::cuda::par.on(s), data.begin(), data.end(), result.begin(), is_even<T>());

  REQUIRE(end - result.begin() == 3);
  result.erase(end, result.end());

  const Vector ref{1, 1, 3};
  REQUIRE(result == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestRemoveCopyIfStencilCudaStreams", "[remove]")
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector stencil{0, 1, 0, 0, 1};

  Vector result(5);

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end = thrust::remove_copy_if(
    thrust::cuda::par.on(s), data.begin(), data.end(), stencil.begin(), result.begin(), ::cuda::std::identity{});

  REQUIRE(end - result.begin() == 3);
  result.erase(end, result.end());

  const Vector ref{1, 1, 3};
  REQUIRE(result == ref);

  cudaStreamDestroy(s);
}
