#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_system& system, RandomAccessIterator1, RandomAccessIterator1, RandomAccessIterator2)
{
  system.validate_dispatch();
}

void TestStableSortByKeyDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::stable_sort_by_key(sys, vec.begin(), vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestStableSortByKeyDispatchExplicit);

template <typename RandomAccessIterator1, typename RandomAccessIterator2>
void stable_sort_by_key(my_tag, RandomAccessIterator1 keys_first, RandomAccessIterator1, RandomAccessIterator2)
{
  *keys_first = 13;
}

void TestStableSortByKeyDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::stable_sort_by_key(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStableSortByKeyDispatchImplicit);

template <typename T>
struct less_div_10
{
  _CCCL_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return ((int) lhs) / 10 < ((int) rhs) / 10;
  }
};

template <class Vector>
void InitializeSimpleStableKeyValueSortTest(
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
void TestStableSortByKeySimple()
{
  using T = typename Vector::value_type;

  Vector unsorted_keys, unsorted_values;
  Vector sorted_keys, sorted_values;

  InitializeSimpleStableKeyValueSortTest(unsorted_keys, unsorted_values, sorted_keys, sorted_values);

  thrust::stable_sort_by_key(unsorted_keys.begin(), unsorted_keys.end(), unsorted_values.begin(), less_div_10<T>());

  ASSERT_EQUAL(unsorted_keys, sorted_keys);
  ASSERT_EQUAL(unsorted_values, sorted_values);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestStableSortByKeySimple);

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

    ASSERT_EQUAL(h_keys, d_keys);
    ASSERT_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestStableSortByKey, SignedIntegralTypes> TestStableSortByKeyInstance;

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

    ASSERT_EQUAL(h_keys, d_keys);
    ASSERT_EQUAL(h_values, d_values);
  }
};
VariableUnitTest<TestStableSortByKeySemantics,
                 unittest::type_list<unittest::uint8_t, unittest::uint16_t, unittest::uint32_t>>
  TestStableSortByKeySemanticsInstance;
