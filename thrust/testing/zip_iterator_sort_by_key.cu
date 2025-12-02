#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename T>
struct TestZipIteratorStableSortByKey
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h1 = unittest::random_integers<T>(n);
    thrust::host_vector<T> h2 = unittest::random_integers<T>(n);
    thrust::host_vector<T> h3 = unittest::random_integers<T>(n);
    thrust::host_vector<T> h4 = unittest::random_integers<T>(n);

    thrust::device_vector<T> d1 = h1;
    thrust::device_vector<T> d2 = h2;
    thrust::device_vector<T> d3 = h3;
    thrust::device_vector<T> d4 = h4;

    // sort with (tuple, scalar)
    thrust::stable_sort_by_key(
      thrust::make_zip_iterator(h1.begin(), h2.begin()), thrust::make_zip_iterator(h1.end(), h2.end()), h3.begin());
    thrust::stable_sort_by_key(
      thrust::make_zip_iterator(d1.begin(), d2.begin()), thrust::make_zip_iterator(d1.end(), d2.end()), d3.begin());

    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
    ASSERT_EQUAL_QUIET(h3, d3);
    ASSERT_EQUAL_QUIET(h4, d4);

    // sort with (scalar, tuple)
    thrust::stable_sort_by_key(h1.begin(), h1.end(), thrust::make_zip_iterator(h3.begin(), h4.begin()));
    thrust::stable_sort_by_key(d1.begin(), d1.end(), thrust::make_zip_iterator(d3.begin(), d4.begin()));

    // sort with (tuple, tuple)
    thrust::stable_sort_by_key(thrust::make_zip_iterator(h1.begin(), h2.begin()),
                               thrust::make_zip_iterator(h1.end(), h2.end()),
                               thrust::make_zip_iterator(h3.begin(), h4.begin()));
    thrust::stable_sort_by_key(thrust::make_zip_iterator(d1.begin(), d2.begin()),
                               thrust::make_zip_iterator(d1.end(), d2.end()),
                               thrust::make_zip_iterator(d3.begin(), d4.begin()));

    ASSERT_EQUAL_QUIET(h1, d1);
    ASSERT_EQUAL_QUIET(h2, d2);
    ASSERT_EQUAL_QUIET(h3, d3);
    ASSERT_EQUAL_QUIET(h4, d4);
  }
};
VariableUnitTest<TestZipIteratorStableSortByKey,
                 unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestZipIteratorStableSortByKeyInstance;
