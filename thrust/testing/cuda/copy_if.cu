#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

#include "thrust/iterator/transform_iterator.h"
#include <unittest/unittest.h>

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return (static_cast<unsigned int>(x) & 1) == 0;
  }
};

template <typename T>
struct mod_3
{
  _CCCL_HOST_DEVICE unsigned int operator()(T x)
  {
    return static_cast<unsigned int>(x) % 3;
  }
};

template <typename T>
struct mod_n
{
  T mod;
  _CCCL_HOST_DEVICE bool operator()(T x)
  {
    return (x % mod == 0) ? true : false;
  }
};

template <typename T>
struct multiply_n
{
  T multiplier;
  _CCCL_HOST_DEVICE T operator()(T x)
  {
    return x * multiplier;
  }
};

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator1, typename Iterator2, typename Predicate, typename Iterator3>
__global__ void copy_if_kernel(
  ExecutionPolicy exec, Iterator1 first, Iterator1 last, Iterator2 result1, Predicate pred, Iterator3 result2)
{
  *result2 = thrust::copy_if(exec, first, last, result1, pred);
}

template <typename ExecutionPolicy>
void TestCopyIfDevice(ExecutionPolicy exec)
{
  size_t n                          = 1000;
  thrust::host_vector<int> h_data   = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;

  typename thrust::host_vector<int>::iterator h_new_end;
  typename thrust::device_vector<int>::iterator d_new_end;

  thrust::device_vector<typename thrust::device_vector<int>::iterator> d_new_end_vec(1);

  // test with Predicate that returns a bool
  {
    thrust::host_vector<int> h_result(n);
    thrust::device_vector<int> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());

    copy_if_kernel<<<1, 1>>>(
      exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);

    d_new_end = d_new_end_vec[0];

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    REQUIRE(h_result == d_result);
  }

  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<int> h_result(n);
    thrust::device_vector<int> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<int>());

    copy_if_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);

    d_new_end = d_new_end_vec[0];

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    REQUIRE(h_result == d_result);
  }
}

TEST_CASE("TestCopyIfDeviceSeq", "[copy_if]")
{
  TestCopyIfDevice(thrust::seq);
}

TEST_CASE("TestCopyIfDeviceDevice", "[copy_if]")
{
  TestCopyIfDevice(thrust::device);
}

TEST_CASE("TestCopyIfDeviceNoSync", "[copy_if]")
{
  TestCopyIfDevice(thrust::cuda::par_nosync);
}
#endif

template <typename ExecutionPolicy>
void TestCopyIfCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;

  Vector data{1, 2, 1, 3, 2};
  Vector result(data.size());

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end = thrust::copy_if(policy.on(s), data.begin(), data.end(), result.begin(), is_even<int>());

  REQUIRE(end - result.begin() == 2);
  result.resize(end - result.begin());
  const Vector ref{2, 2};
  REQUIRE(result == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestCopyIfCudaStreamsSync", "[copy_if]")
{
  TestCopyIfCudaStreams(thrust::cuda::par);
}

TEST_CASE("TestCopyIfCudaStreamsNoSync", "[copy_if]")
{
  TestCopyIfCudaStreams(thrust::cuda::par_nosync);
}

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy,
          typename Iterator1,
          typename Iterator2,
          typename Iterator3,
          typename Predicate,
          typename Iterator4>
__global__ void copy_if_kernel(
  ExecutionPolicy exec,
  Iterator1 first,
  Iterator1 last,
  Iterator2 stencil_first,
  Iterator3 result1,
  Predicate pred,
  Iterator4 result2)
{
  *result2 = thrust::copy_if(exec, first, last, stencil_first, result1, pred);
}

template <typename ExecutionPolicy>
void TestCopyIfStencilDevice(ExecutionPolicy exec)
{
  size_t n = 1000;
  thrust::host_vector<int> h_data(n);
  thrust::sequence(h_data.begin(), h_data.end());
  thrust::device_vector<int> d_data(n);
  thrust::sequence(d_data.begin(), d_data.end());

  thrust::host_vector<int> h_stencil   = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_stencil = unittest::random_integers<int>(n);

  typename thrust::host_vector<int>::iterator h_new_end;
  typename thrust::device_vector<int>::iterator d_new_end;

  thrust::device_vector<typename thrust::device_vector<int>::iterator> d_new_end_vec(1);

  // test with Predicate that returns a bool
  {
    thrust::host_vector<int> h_result(n);
    thrust::device_vector<int> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), is_even<int>());

    copy_if_kernel<<<1, 1>>>(
      exec, d_data.begin(), d_data.end(), d_result.begin(), is_even<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);

    d_new_end = d_new_end_vec[0];

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    REQUIRE(h_result == d_result);
  }

  // test with Predicate that returns a non-bool
  {
    thrust::host_vector<int> h_result(n);
    thrust::device_vector<int> d_result(n);

    h_new_end = thrust::copy_if(h_data.begin(), h_data.end(), h_result.begin(), mod_3<int>());

    copy_if_kernel<<<1, 1>>>(exec, d_data.begin(), d_data.end(), d_result.begin(), mod_3<int>(), d_new_end_vec.begin());
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);

    d_new_end = d_new_end_vec[0];

    h_result.resize(h_new_end - h_result.begin());
    d_result.resize(d_new_end - d_result.begin());

    REQUIRE(h_result == d_result);
  }
}

TEST_CASE("TestCopyIfStencilDeviceSeq", "[copy_if]")
{
  TestCopyIfStencilDevice(thrust::seq);
}

TEST_CASE("TestCopyIfStencilDeviceDevice", "[copy_if]")
{
  TestCopyIfStencilDevice(thrust::device);
}

TEST_CASE("TestCopyIfStencilDeviceNoSync", "[copy_if]")
{
  TestCopyIfStencilDevice(thrust::cuda::par_nosync);
}
#endif

template <typename ExecutionPolicy>
void TestCopyIfStencilCudaStreams(ExecutionPolicy policy)
{
  using Vector = thrust::device_vector<int>;
  using T      = Vector::value_type;

  Vector data{1, 2, 1, 3, 2};

  Vector result(5);

  Vector stencil{0, 1, 0, 0, 1};

  cudaStream_t s;
  cudaStreamCreate(&s);

  const Vector::iterator end =
    thrust::copy_if(policy.on(s), data.begin(), data.end(), stencil.begin(), result.begin(), ::cuda::std::identity{});

  REQUIRE(end - result.begin() == 2);
  result.resize(end - result.begin());

  const Vector ref{2, 2};
  REQUIRE(result == ref);

  cudaStreamDestroy(s);
}

TEST_CASE("TestCopyIfStencilCudaStreamsSync", "[copy_if]")
{
  TestCopyIfStencilCudaStreams(thrust::cuda::par);
}

TEST_CASE("TestCopyIfStencilCudaStreamsNoSync", "[copy_if]")
{
  TestCopyIfStencilCudaStreams(thrust::cuda::par_nosync);
}

void TestCopyIfWithMagnitude(int magnitude)
{
  using offset_t = std::size_t;

  // Prepare input
  const offset_t num_items = offset_t{1ull} << magnitude;
  const thrust::counting_iterator<offset_t> begin(offset_t{0});
  auto end = begin + static_cast<std::ptrdiff_t>(num_items);
  REQUIRE(static_cast<offset_t>(::cuda::std::distance(begin, end)) == num_items);

  // Run algorithm on large number of items
  const offset_t match_every_nth     = 1000000;
  const offset_t expected_num_copied = (num_items + match_every_nth - 1) / match_every_nth;
  thrust::device_vector<offset_t> copied_out(expected_num_copied);
  auto selected_out_end = thrust::copy_if(begin, end, copied_out.begin(), mod_n<offset_t>{match_every_nth});

  // Ensure number of selected items are correct
  const offset_t num_selected_out = static_cast<offset_t>(::cuda::std::distance(copied_out.begin(), selected_out_end));
  REQUIRE(num_selected_out == expected_num_copied);
  copied_out.resize(expected_num_copied);

  // Ensure selected items are correct
  auto expected_out_it           = thrust::make_transform_iterator(begin, multiply_n<offset_t>{match_every_nth});
  const bool all_results_correct = thrust::equal(copied_out.begin(), copied_out.end(), expected_out_it);
  REQUIRE(all_results_correct);
}

TEST_CASE("TestCopyIfWithLargeNumberOfItems", "[copy_if]")
{
  TestCopyIfWithMagnitude(30);
  TestCopyIfWithMagnitude(31);
  TestCopyIfWithMagnitude(32);
  TestCopyIfWithMagnitude(33);
}

void TestCopyIfStencilWithMagnitude(int magnitude)
{
  using offset_t = std::size_t;

  // Prepare input
  const offset_t num_items = offset_t{1ull} << magnitude;
  const thrust::counting_iterator<offset_t> begin(offset_t{0});
  auto end = begin + static_cast<std::ptrdiff_t>(num_items);
  const thrust::counting_iterator<offset_t> stencil(offset_t{0});
  REQUIRE(static_cast<offset_t>(::cuda::std::distance(begin, end)) == num_items);

  // Run algorithm on large number of items
  const offset_t match_every_nth     = 1000000;
  const offset_t expected_num_copied = (num_items + match_every_nth - 1) / match_every_nth;
  thrust::device_vector<offset_t> copied_out(expected_num_copied);
  auto selected_out_end = thrust::copy_if(begin, end, stencil, copied_out.begin(), mod_n<offset_t>{match_every_nth});

  // Ensure number of selected items are correct
  const offset_t num_selected_out = static_cast<offset_t>(::cuda::std::distance(copied_out.begin(), selected_out_end));
  REQUIRE(num_selected_out == expected_num_copied);
  copied_out.resize(expected_num_copied);

  // Ensure selected items are correct
  auto expected_out_it           = thrust::make_transform_iterator(begin, multiply_n<offset_t>{match_every_nth});
  const bool all_results_correct = thrust::equal(copied_out.begin(), copied_out.end(), expected_out_it);
  REQUIRE(all_results_correct);
}

TEST_CASE("TestCopyIfStencilWithLargeNumberOfItems", "[copy_if]")
{
  TestCopyIfStencilWithMagnitude(30);
  TestCopyIfStencilWithMagnitude(31);
  TestCopyIfStencilWithMagnitude(32);
  TestCopyIfStencilWithMagnitude(33);
}
