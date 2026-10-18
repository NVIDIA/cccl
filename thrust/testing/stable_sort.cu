#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include <unittest/unittest.h>

template <typename RandomAccessIterator>
void stable_sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
  system.validate_dispatch();
}

TEST_CASE("TestStableSortDispatchExplicit", "[stable_sort]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::stable_sort(sys, vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename RandomAccessIterator>
void stable_sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
  *first = 13;
}

TEST_CASE("TestStableSortDispatchImplicit", "[stable_sort]")
{
  thrust::device_vector<int> vec(1);

  thrust::stable_sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

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
void initialize_simple_stable_key_sort_test(Vector& unsorted_keys, Vector& sorted_keys)
{
  unsorted_keys.resize(9);
  unsorted_keys = {25, 14, 35, 16, 26, 34, 36, 24, 15};

  sorted_keys.resize(9);
  sorted_keys = {14, 16, 15, 25, 26, 24, 35, 34, 36};
}

template <class Vector>
void test_stable_sort_simple()
{
  using T = typename Vector::value_type;

  Vector unsorted_keys;
  Vector sorted_keys;

  initialize_simple_stable_key_sort_test(unsorted_keys, sorted_keys);

  thrust::stable_sort(unsorted_keys.begin(), unsorted_keys.end(), less_div_10<T>());

  REQUIRE(unsorted_keys == sorted_keys);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_stable_sort_simple);

template <typename T>
struct TestStableSort
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    REQUIRE(h_data == d_data);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestStableSort, SignedIntegralTypes);

template <typename T>
struct TestStableSortSemantics
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    thrust::stable_sort(h_data.begin(), h_data.end(), less_div_10<T>());
    thrust::stable_sort(d_data.begin(), d_data.end(), less_div_10<T>());

    REQUIRE(h_data == d_data);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestStableSortSemantics,
                                          unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>);

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
void test_stable_sort_with_indirection()
{
  // add numbers modulo 3 with external lookup table
  using T = typename Vector::value_type;

  Vector data{1, 3, 5, 3, 0, 2, 1};
  Vector table{0, 1, 2, 0, 1, 2};

  thrust::stable_sort(data.begin(), data.end(), comp_mod3<T>(thrust::raw_pointer_cast(&table[0])));

  Vector ref{3, 3, 0, 1, 1, 5, 2};
  REQUIRE(data == ref);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(test_stable_sort_with_indirection);

// -0.0 and +0.0 compare equal, so a stable sort has to keep them in their input order (see #750)
template <typename T, typename Compare>
void test_stable_sort_signed_zeros(Compare comp)
{
  const int n = 10000;
  thrust::host_vector<T> h_keys(n);
  for (int i = 0; i < n; ++i)
  {
    h_keys[i] = i % 3 == 0 ? T(0.0) : i % 3 == 1 ? T(-0.0) : T(i % 7) - T(3.0);
  }

  std::vector<T> expected(h_keys.begin(), h_keys.end());
  std::stable_sort(expected.begin(), expected.end(), comp);

  thrust::device_vector<T> d_keys = h_keys;
  thrust::stable_sort(h_keys.begin(), h_keys.end(), comp);
  thrust::stable_sort(d_keys.begin(), d_keys.end(), comp);
  const thrust::host_vector<T> d_result = d_keys;

  bool host_matches   = true;
  bool device_matches = true;
  for (int i = 0; i < n; ++i)
  {
    host_matches &= h_keys[i] == expected[i] && std::signbit(h_keys[i]) == std::signbit(expected[i]);
    device_matches &= d_result[i] == expected[i] && std::signbit(d_result[i]) == std::signbit(expected[i]);
  }
  REQUIRE(host_matches);
  REQUIRE(device_matches);
}

TEST_CASE("TestStableSortSignedZeros", "[stable_sort]")
{
  test_stable_sort_signed_zeros<float>(thrust::less<float>());
  test_stable_sort_signed_zeros<float>(thrust::greater<float>());
  test_stable_sort_signed_zeros<double>(thrust::less<double>());
  test_stable_sort_signed_zeros<double>(thrust::greater<double>());
}
