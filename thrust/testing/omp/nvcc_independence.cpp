#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system_error.h>
#include <thrust/transform.h>

#include <unittest/unittest.h>

TEST_CASE("TestNvccIndependenceTransform", "[nvcc_independence]")
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::transform(h_input.begin(), h_input.end(), h_output.begin(), ::cuda::std::negate<T>());
  thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), ::cuda::std::negate<T>());

  REQUIRE(h_output == d_output);
}

TEST_CASE("TestNvccIndependenceReduce", "[nvcc_independence]")
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  T init = 13;

  T h_result = thrust::reduce(h_data.begin(), h_data.end(), init);
  T d_result = thrust::reduce(d_data.begin(), d_data.end(), init);

  ASSERT_ALMOST_EQUAL(h_result, d_result);
}

TEST_CASE("TestNvccIndependenceExclusiveScan", "[nvcc_independence]")
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_input   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_input = h_input;

  thrust::host_vector<T> h_output(n);
  thrust::device_vector<T> d_output(n);

  thrust::inclusive_scan(h_input.begin(), h_input.end(), h_output.begin());
  thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
  REQUIRE(d_output == h_output);
}

TEST_CASE("TestNvccIndependenceSort", "[nvcc_independence]")
{
  using T     = int;
  const int n = 10;

  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), ::cuda::std::less<T>());
  thrust::sort(d_data.begin(), d_data.end(), ::cuda::std::less<T>());

  REQUIRE(h_data == d_data);
}
