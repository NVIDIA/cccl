#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/partition.h>

#include <unittest/unittest.h>

#ifdef THRUST_TEST_DEVICE_SIDE
template <typename ExecutionPolicy, typename Iterator, typename Predicate, typename Iterator2>
__global__ void
is_partitioned_kernel(ExecutionPolicy exec, Iterator first, Iterator last, Predicate pred, Iterator2 result)
{
  *result = thrust::is_partitioned(exec, first, last, pred);
}

template <typename T>
struct is_even
{
  _CCCL_HOST_DEVICE bool operator()(T x) const
  {
    return ((int) x % 2) == 0;
  }
};

template <typename ExecutionPolicy>
void TestIsPartitionedDevice(ExecutionPolicy exec)
{
  size_t n = 1000;

  n = ::cuda::std::max<size_t>(n, 2);

  thrust::device_vector<int> v = unittest::random_integers<int>(n);

  thrust::device_vector<bool> result(1);

  v[0] = 1;
  v[1] = 0;

  is_partitioned_kernel<<<1, 1>>>(exec, v.begin(), v.end(), is_even<int>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE_FALSE(result[0]);

  thrust::partition(v.begin(), v.end(), is_even<int>());

  is_partitioned_kernel<<<1, 1>>>(exec, v.begin(), v.end(), is_even<int>(), result.begin());
  {
    cudaError_t const err = cudaDeviceSynchronize();
    REQUIRE(cudaSuccess == err);
  }

  REQUIRE(result[0]);
}

TEST_CASE("TestIsPartitionedDeviceSeq", "[is_partitioned]")
{
  TestIsPartitionedDevice(thrust::seq);
}

TEST_CASE("TestIsPartitionedDeviceDevice", "[is_partitioned]")
{
  TestIsPartitionedDevice(thrust::device);
}
#endif

TEST_CASE("TestIsPartitionedCudaStreams", "[is_partitioned]")
{
  thrust::device_vector<int> v(4);
  v[0] = 1;
  v[1] = 1;
  v[2] = 1;
  v[3] = 0;

  cudaStream_t s;
  cudaStreamCreate(&s);

  // empty partition
  REQUIRE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin(), ::cuda::std::identity{}));

  // one element true partition
  REQUIRE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin() + 1, ::cuda::std::identity{}));

  // just true partition
  REQUIRE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.begin() + 2, ::cuda::std::identity{}));

  // both true & false partitions
  REQUIRE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.end(), ::cuda::std::identity{}));

  // one element false partition
  REQUIRE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin() + 3, v.end(), ::cuda::std::identity{}));

  v[0] = 1;
  v[1] = 0;
  v[2] = 1;
  v[3] = 1;

  // not partitioned
  REQUIRE_FALSE(thrust::is_partitioned(thrust::cuda::par.on(s), v.begin(), v.end(), ::cuda::std::identity{}));

  cudaStreamDestroy(s);
}

template <typename T>
struct is_even_non_const
{
  _CCCL_HOST_DEVICE bool operator()(T x) // no const
  {
    return ((int) x % 2) == 0;
  }
};

TEST_CASE("TestIsPartitionedWithNonConstPredicate", "[is_partitioned]")
{
  thrust::device_vector<int> partitioned   = {0, 2, 4, 1, 3, 5};
  thrust::device_vector<int> unpartitioned = {0, 1, 2, 3};

  REQUIRE(thrust::is_partitioned(thrust::cuda::par, partitioned.begin(), partitioned.end(), is_even_non_const<int>{}));

  REQUIRE_FALSE(
    thrust::is_partitioned(thrust::cuda::par, unpartitioned.begin(), unpartitioned.end(), is_even_non_const<int>{}));
}
