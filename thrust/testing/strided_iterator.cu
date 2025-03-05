#include <thrust/device_vector.h>
#include <thrust/iterator/strided_iterator.h>

#include <cuda/std/__numeric/iota.h>
#include <cuda/std/array>
#include <cuda/std/utility>

#include <unittest/unittest.h>

void TestStridedIterator()
{
  // iterate over all second elements (runtime stride)
  {
    thrust::device_vector<int> v(10);
    auto iter = thrust::make_strided_iterator(v.begin(), 2);
    cuda::std::fill(iter, iter + 3, 42);
    ASSERT_EQUAL(v, (thrust::device_vector{42, 0, 42, 0, 42, 0, 0, 0, 0, 0}));
  }

  // iterate over all second elements (static stride)
  {
    thrust::device_vector<int> v(10);
    auto iter = thrust::make_strided_iterator<2>(v.begin());
    cuda::std::fill(iter, iter + 3, 42);
    ASSERT_EQUAL(v, (thrust::device_vector{42, 0, 42, 0, 42, 0, 0, 0, 0, 0}));
  }
}
DECLARE_UNITTEST(TestStridedIterator);

void TestStridedIteratorStruct()
{
  using arr_of_pairs   = ::cuda::std::array<::cuda::std::pair<int, double>, 4>;
  const auto reference = arr_of_pairs{{{1, 1337}, {3, 1337}, {5, 1337}, {7, 1337}}};

  // iterate over all second elements (runtime stride)
  {
    auto arr  = arr_of_pairs{{{1, 2}, {3, 4}, {5, 6}, {7, 8}}};
    auto iter = thrust::make_strided_iterator(&arr[0].second, sizeof(::cuda::std::pair<int, double>));

    cuda::std::fill(iter, iter + 4, 1337);

    ASSERT_EQUAL(arr == reference, true);
  }

  // iterate over all second elements (static stride)
  {
    auto arr  = ::cuda::std::array<::cuda::std::pair<int, double>, 4>{{{1, 2}, {3, 4}, {5, 6}, {7, 8}}};
    auto iter = thrust::make_strided_iterator<sizeof(::cuda::std::pair<int, double>)>(&arr[0].second);

    cuda::std::fill(iter, iter + 4, 1337);

    ASSERT_EQUAL(arr == reference, true);
  }
}
DECLARE_UNITTEST(TestStridedIteratorStruct);
