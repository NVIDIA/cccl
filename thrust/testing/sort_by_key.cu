#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_system& system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
  system.validate_dispatch();
}

TEST_CASE("TestSortByKeyDispatchExplicit", "[sort_by_key]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
  *keys_first = 13;
}

TEST_CASE("TestSortByKeyDispatchImplicit", "[sort_by_key]")
{
  thrust::device_vector<int> vec(1);

  thrust::sort_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <class Vector>
void initialize_simple_key_value_sort_test(
  Vector& unsorted_keys, Vector& unsorted_values, Vector& sorted_keys, Vector& sorted_values)
{
  unsorted_keys.resize(7);
  unsorted_keys = {1, 3, 6, 5, 2, 0, 4};
  unsorted_values.resize(7);
  unsorted_values = {0, 1, 2, 3, 4, 5, 6};

  sorted_keys.resize(7);
  sorted_keys = {0, 1, 2, 3, 4, 5, 6};
  sorted_values.resize(7);
  sorted_values = {5, 0, 4, 1, 6, 3, 2};
}

template <class Vector>
void test_sort_by_key_simple()
{
  Vector unsorted_keys, unsorted_values;
  Vector sorted_keys, sorted_values;

  initialize_simple_key_value_sort_test(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

  thrust::sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin());

  REQUIRE(unsorted_keys == sorted_keys);
  REQUIRE(unsorted_values == sorted_values);
}
DECLARE_VECTOR_UNITTEST(test_sort_by_key_simple);

template <typename T>
void test_sort_ascending_key_value(const size_t n)
{
  thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_keys = h_keys;

  thrust::host_vector<T> h_values   = h_keys;
  thrust::device_vector<T> d_values = d_keys;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::less<T>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::less<T>());

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_values == d_values);
}
DECLARE_VARIABLE_UNITTEST(test_sort_ascending_key_value);

template <typename T>
void test_sort_descending_key_value(const size_t n)
{
  thrust::host_vector<int> h_keys   = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_keys = h_keys;

  thrust::host_vector<int> h_values   = h_keys;
  thrust::device_vector<int> d_values = d_keys;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::greater<int>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::greater<int>());

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_values == d_values);
}
DECLARE_VARIABLE_UNITTEST(test_sort_descending_key_value);

TEST_CASE("TestSortByKeyBool", "[sort_by_key]")
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_keys  = unittest::random_integers<bool>(n);
  thrust::host_vector<int> h_values = unittest::random_integers<int>(n);

  thrust::device_vector<bool> d_keys  = h_keys;
  thrust::device_vector<int> d_values = h_values;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_values == d_values);
}

TEST_CASE("TestSortByKeyBoolDescending", "[sort_by_key]")
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_keys  = unittest::random_integers<bool>(n);
  thrust::host_vector<int> h_values = unittest::random_integers<int>(n);

  thrust::device_vector<bool> d_keys  = h_keys;
  thrust::device_vector<int> d_values = h_values;

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), ::cuda::std::greater<bool>());
  thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), ::cuda::std::greater<bool>());

  REQUIRE(h_keys == d_keys);
  REQUIRE(h_values == d_values);
}

TEST_CASE("TestSortByKeyLongDouble", "[sort_by_key]")
{
  thrust::host_vector<long double> h_keys          = {10.0L, 9.0L, 8.0L, 7.0L, 6.0L, 5.0L, 4.0L, 3.0L, 2.0L, 1.0L};
  thrust::host_vector<int> h_values                = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const thrust::host_vector<int> h_values_expected = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

  thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());

  REQUIRE(thrust::is_sorted(h_keys.begin(), h_keys.end()));
  REQUIRE(h_values == h_values_expected);
}
