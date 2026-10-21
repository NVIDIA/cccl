#include <thrust/functional.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename RandomAccessIterator>
void sort(my_system& system, RandomAccessIterator, RandomAccessIterator)
{
  system.validate_dispatch();
}

TEST_CASE("TestSortDispatchExplicit", "[sort]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::sort(sys, vec.begin(), vec.begin());

  REQUIRE(sys.is_valid());
}

template <typename RandomAccessIterator>
void sort(my_tag, RandomAccessIterator first, RandomAccessIterator)
{
  *first = 13;
}

TEST_CASE("TestSortDispatchImplicit", "[sort]")
{
  thrust::device_vector<int> vec(1);

  thrust::sort(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}

template <class Vector>
void initialize_simple_key_sort_test(Vector& unsorted_keys, Vector& sorted_keys)
{
  unsorted_keys.resize(7);
  unsorted_keys = {1, 3, 6, 5, 2, 0, 4};

  sorted_keys.resize(7);
  sorted_keys = {0, 1, 2, 3, 4, 5, 6};
}

template <class Vector>
void test_sort_simple()
{
  Vector unsorted_keys;
  Vector sorted_keys;

  initialize_simple_key_sort_test(unsorted_keys, sorted_keys);

  thrust::sort(unsorted_keys.begin(), unsorted_keys.end());

  REQUIRE(unsorted_keys == sorted_keys);
}
DECLARE_VECTOR_UNITTEST(test_sort_simple);

template <typename T>
void test_sort_ascending_key(const size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), ::cuda::std::less<T>());
  thrust::sort(d_data.begin(), d_data.end(), ::cuda::std::less<T>());

  REQUIRE(h_data == d_data);
}
DECLARE_VARIABLE_UNITTEST(test_sort_ascending_key);

TEST_CASE("TestSortDescendingKey", "[sort]")
{
  const size_t n = 10027;

  thrust::host_vector<int> h_data   = unittest::random_integers<int>(n);
  thrust::device_vector<int> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), ::cuda::std::greater<int>());
  thrust::sort(d_data.begin(), d_data.end(), ::cuda::std::greater<int>());

  REQUIRE(h_data == d_data);
}

TEST_CASE("TestSortBool", "[sort]")
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_data   = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end());
  thrust::sort(d_data.begin(), d_data.end());

  REQUIRE(h_data == d_data);
}

TEST_CASE("TestSortBoolDescending", "[sort]")
{
  const size_t n = 10027;

  thrust::host_vector<bool> h_data   = unittest::random_integers<bool>(n);
  thrust::device_vector<bool> d_data = h_data;

  thrust::sort(h_data.begin(), h_data.end(), ::cuda::std::greater<bool>());
  thrust::sort(d_data.begin(), d_data.end(), ::cuda::std::greater<bool>());

  REQUIRE(h_data == d_data);
}

// See also: https://github.com/NVIDIA/cccl/issues/4919
TEST_CASE("TestSortTrivial", "[sort]")
{
  thrust::host_vector<int> h_data    = {1, 0, -1, -2, -3};
  const thrust::host_vector<int> ref = {-3, -2, -1, 0, 1};

  thrust::sort(h_data.begin(), h_data.end());
  REQUIRE(h_data == ref);
}
