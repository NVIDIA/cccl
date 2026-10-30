#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>

#include <unittest/unittest.h>

// see also: https://github.com/NVIDIA/cccl/issues/3541
void test_transform_with_lambda()
{
  auto l = [] __host__ __device__(int v) {
    return v < 4;
  };
  thrust::host_vector<int> A{1, 2, 3, 4, 5, 6, 7};
  REQUIRE(thrust::any_of(A.begin(), A.end(), l));

  thrust::device_vector<int> B{1, 2, 3, 4, 5, 6, 7};
  REQUIRE(thrust::any_of(B.begin(), B.end(), l));
}
TEST_CASE("test_transform_with_lambda", "[transform_iterator]")
{
  test_transform_with_lambda();
}
