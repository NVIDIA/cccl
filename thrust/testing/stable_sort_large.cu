#include <thrust/functional.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename T, unsigned int N>
void _TestStableSortWithLargeKeys()
{
  size_t n = (128 * 1024) / sizeof(FixedVector<T, N>);

  thrust::host_vector<FixedVector<T, N>> h_keys(n);

  const thrust::host_vector<T> values = unittest::random_integers<T>(n);
  for (size_t i = 0; i < n; i++)
  {
    h_keys[i] = FixedVector<T, N>(values[i]);
  }

  thrust::device_vector<FixedVector<T, N>> d_keys = h_keys;

  thrust::stable_sort(h_keys.begin(), h_keys.end());
  thrust::stable_sort(d_keys.begin(), d_keys.end());

  ASSERT_EQUAL_QUIET(h_keys, d_keys);
}

void TestStableSortWithLargeKeys()
{
  _TestStableSortWithLargeKeys<int, 2>();
  _TestStableSortWithLargeKeys<int, 17>();
  _TestStableSortWithLargeKeys<int, 128>();
}
DECLARE_UNITTEST(TestStableSortWithLargeKeys);
