#include <thrust/reduce.h>

#include <unittest/unittest.h>

template <typename T, unsigned int N>
void test_reduce_with_large_types()
{
  const size_t n = (64 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_data(n);

  for (size_t i = 0; i < h_data.size(); i++)
  {
    h_data[i] = FixedVector<T, N>(static_cast<T>(i));
  }

  thrust::device_vector<FixedVector<T, N>> d_data = h_data;

  const FixedVector<T, N> h_result = thrust::reduce(h_data.begin(), h_data.end(), FixedVector<T, N>(T{0}));
  const FixedVector<T, N> d_result = thrust::reduce(d_data.begin(), d_data.end(), FixedVector<T, N>(T{0}));

  REQUIRE(h_result == d_result);
}

TEST_CASE("TestReduceWithLargeTypes", "[reduce_large]")
{
  test_reduce_with_large_types<int, 4>();
  test_reduce_with_large_types<int, 8>();
  test_reduce_with_large_types<int, 16>();

  // XXX these take too long to compile
  //  test_reduce_with_large_types<int,   32>();
  //  test_reduce_with_large_types<int,   64>();
  //  test_reduce_with_large_types<int,  128>();
  //  test_reduce_with_large_types<int,  256>();
  //  test_reduce_with_large_types<int,  512>();
}
