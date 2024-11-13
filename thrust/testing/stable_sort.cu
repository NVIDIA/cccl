#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator>
void stable_sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
  system.validate_dispatch();
}

void TestStableSortDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::stable_sort(sys, vec.begin(), vec.begin());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestStableSortDispatchExplicit);

template <typename RandomAccessIterator>
void stable_sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
  *first = 13;
}

void TestStableSortDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::stable_sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestStableSortDispatchImplicit);

template <typename T>
struct less_div_10
{
  _CCCL_HOST_DEVICE bool operator()(const T& lhs, const T& rhs) const
  {
    return ((int) lhs) / 10 < ((int) rhs) / 10;
  }
};

template <class Vector>
void InitializeSimpleStableKeySortTest(Vector& unsorted_keys, Vector& sorted_keys)
{
  unsorted_keys.resize(9);
  unsorted_keys = {25, 14, 35, 16, 26, 34, 36, 24, 15};

  sorted_keys.resize(9);
  sorted_keys = {14, 16, 15, 25, 26, 24, 35, 34, 36};
}

template <class Vector>
void TestStableSortSimple()
{
  using T = typename Vector::value_type;

  Vector unsorted_keys;
  Vector sorted_keys;

  InitializeSimpleStableKeySortTest(unsorted_keys, sorted_keys);

  thrust::stable_sort(unsorted_keys.begin(), unsorted_keys.end(), less_div_10<T>());

  ASSERT_EQUAL(unsorted_keys, sorted_keys);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestStableSortSimple);

template <typename T>
struct TestStableSort
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    ASSERT_EQUAL(h_data, d_data);
  }
};
VariableUnitTest<TestStableSort, SignedIntegralTypes> TestStableSortInstance;

template <typename T>
struct TestStableSortSemantics
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    ASSERT_EQUAL(h_data, d_data);
  }
};
VariableUnitTest<TestStableSortSemantics, unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>>
  TestStableSortSemanticsInstance;

template <typename T>
struct comp_mod3
{
  T* table;

  comp_mod3(T* table)
      : table(table)
  {}

  _CCCL_HOST_DEVICE bool operator()(T a, T b)
  {
    return table[(int) a] < table[(int) b];
  }
};

template <typename Vector>
void TestStableSortWithIndirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{1, 3, 5, 3, 0, 2, 1};
  Vector table{0, 1, 2, 0, 1, 2};

  thrust::stable_sort(data.begin(), data.end(), comp_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  Vector ref{3, 3, 0, 1, 1, 5, 2};
  ASSERT_EQUAL(data, ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestStableSortWithIndirection);
