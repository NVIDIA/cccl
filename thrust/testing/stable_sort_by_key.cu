#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_system& system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
  system.validate_dispatch();
}

TEST_CASE("TestStableSortByKeyDispatchExplicit", "[stable_sort_by_key]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::stable_sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
  *keys_first = 13;
}

TEST_CASE("TestStableSortByKeyDispatchImplicit", "[stable_sort_by_key]")
{
  thrust::device_vector<int> vec(1);

  thrust::stable_sort_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <typename T>
struct less_div_10
{
  _CCCL_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return ((int) lhs) / 10 < ((int) rhs) / 10;
  }
};

template <class Vector>
void initialize_simple_stable_key_value_sort_test(
  Vector& unsorted_keys, Vector& unsorted_values, Vector& sorted_keys, Vector& sorted_values)
{
  unsorted_keys.resize(9);
  unsorted_keys = {25, 14, 35, 16, 26, 34, 36, 24, 15};
  unsorted_values.resize(9);
  unsorted_values = {0, 1, 2, 3, 4, 5, 6, 7, 8};

  sorted_keys.resize(9);
  sorted_keys = {14, 16, 15, 25, 26, 24, 35, 34, 36};
  sorted_values.resize(9);
  sorted_values = {1, 3, 8, 0, 4, 7, 2, 5, 6};
}

template <class Vector>
void test_stable_sort_by_key_simple()
{
  using T = typename Vector::value_type;

  Vector unsorted_keys, unsorted_values;
  Vector sorted_keys, sorted_values;

  initialize_simple_stable_key_value_sort_test(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

  thrust::stable_sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin(), less_div_10<T>());

  REQUIRE(unsorted_keys == sorted_keys);
  REQUIRE(unsorted_values == sorted_values);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_stable_sort_by_key_simple);

template <typename T>
struct TestStableSortByKey
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;

    thrust::host_vector<T> h_values   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_values = h_values;

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin());

    REQUIRE(h_keys == d_keys);
    REQUIRE(h_values == d_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestStableSortByKey, SignedIntegralTypes);

template <typename T>
struct TestStableSortByKeySemantics
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_keys   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_keys = h_keys;

    thrust::host_vector<T> h_values   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_values = h_values;

    thrust::stable_sort_by_key(h_keys.begin(), h_keys.end(), h_values.begin(), less_div_10<T>());
    thrust::stable_sort_by_key(d_keys.begin(), d_keys.end(), d_values.begin(), less_div_10<T>());

    REQUIRE(h_keys == d_keys);
    REQUIRE(h_values == d_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(
  TestStableSortByKeySemantics, unittest::type_list<unittest::uint8_t, unittest::uint16_t, unittest::uint32_t>);
