#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/pair.h>
#include <thrust/sequence.h>

#include <unittest/unittest.h>

void TestEqualRangeOnStream()
{ // Regression test for GH issue #921 (nvbug 2173437)
  using vector_t   = typename thrust::device_vector<int>;
  using iterator_t = typename vector_t::iterator;
  using result_t   = thrust::pair<iterator_t, iterator_t>;

  vector_t input(10);
  thrust::sequence(thrust::device, input.begin(), input.end(), 0);
  cudaStream_t stream = 0;
  result_t result     = thrust::equal_range(thrust::cuda::par.on(stream), input.begin(), input.end(), 5);

  ASSERT_EQUAL(5, thrust::distance(input.begin(), result.first));
  ASSERT_EQUAL(6, thrust::distance(input.begin(), result.second));
}
DECLARE_UNITTEST(TestEqualRangeOnStream);
