#include <thrust/iterator/retag.h>
#include <thrust/search.h>
#include <thrust/sequence.h>

#include <cuda/std/cstdint>

#include <unittest/unittest.h>

template <typename T>
struct compare_modulo_pred
{
  T modulus;

  compare_modulo_pred(T modulus)
      : modulus(modulus)
  {}

  _CCCL_HOST_DEVICE bool operator()(T a, T b) const
  {
    return (a % modulus) == (b % modulus);
  }
};

template <class Vector>
void TestSearchSimple()
{
  Vector data{0, 1, 2, 3, 4, 1, 2, 5};
  Vector pattern{1, 2};

  // First occurrence of {1, 2} is at position 1
  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end());
  ASSERT_EQUAL(iter - data.begin(), 1);

  // Pattern not found
  Vector pattern2{9, 9};
  iter = thrust::search(data.begin(), data.end(), pattern2.begin(), pattern2.end());
  ASSERT_EQUAL(iter - data.begin(), static_cast<typename Vector::difference_type>(data.size()));

  // Empty pattern (should return first)
  Vector empty_pattern;
  iter = thrust::search(data.begin(), data.end(), empty_pattern.begin(), empty_pattern.end());
  ASSERT_EQUAL(iter - data.begin(), 0); // Empty pattern found at beginning

  // Pattern longer than data
  Vector long_pattern{1, 2, 3, 4, 5, 6, 7, 8, 9};
  iter = thrust::search(data.begin(), data.end(), long_pattern.begin(), long_pattern.end());
  ASSERT_EQUAL(iter - data.begin(), static_cast<typename Vector::difference_type>(data.size()));

  // Single occurrence
  Vector data2{0, 1, 2, 3, 4, 5};
  Vector pattern3{2, 3};
  iter = thrust::search(data2.begin(), data2.end(), pattern3.begin(), pattern3.end());
  ASSERT_EQUAL(iter - data2.begin(), 2);

  // Multiple occurrences - should find first
  Vector data3{1, 2, 1, 2, 1, 2};
  Vector pattern4{1, 2};
  iter = thrust::search(data3.begin(), data3.end(), pattern4.begin(), pattern4.end());
  ASSERT_EQUAL(iter - data3.begin(), 0); // First occurrence

  // Pattern at the end
  Vector data4{0, 1, 2, 3, 4, 5, 6};
  Vector pattern5{5, 6};
  iter = thrust::search(data4.begin(), data4.end(), pattern5.begin(), pattern5.end());
  ASSERT_EQUAL(iter - data4.begin(), 5);
}
DECLARE_VECTOR_UNITTEST(TestSearchSimple);

template <class Vector>
void TestSearchWithPredicate()
{
  using T = typename Vector::value_type;

  // Test with modulo predicate
  Vector data{0, 1, 3, 2, 4, 5, 7, 6};
  Vector pattern{1, 3}; // Two odd numbers

  // First occurrence of two consecutive odd numbers is at position 1 (1, 3)
  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end(), compare_modulo_pred<T>(2));
  ASSERT_EQUAL(iter - data.begin(), 1);

  // Test with equality predicate
  Vector data2{0, 1, 2, 3, 4, 1, 2, 5};
  Vector pattern2{1, 2};
  iter = thrust::search(data2.begin(), data2.end(), pattern2.begin(), pattern2.end(), ::cuda::std::equal_to<T>());
  ASSERT_EQUAL(iter - data2.begin(), 1);
}
DECLARE_INTEGRAL_VECTOR_UNITTEST(TestSearchWithPredicate);

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1 search(my_system& system, ForwardIterator1 first, ForwardIterator1, ForwardIterator2, ForwardIterator2)
{
  system.validate_dispatch();
  return first;
}

void TestSearchDispatchExplicit()
{
  thrust::device_vector<int> vec(10);
  thrust::device_vector<int> pattern(2);

  my_system sys(0);
  thrust::search(sys, vec.begin(), vec.end(), pattern.begin(), pattern.end());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestSearchDispatchExplicit);

template <typename ForwardIterator1, typename ForwardIterator2>
ForwardIterator1 search(my_tag, ForwardIterator1 first, ForwardIterator1, ForwardIterator2, ForwardIterator2)
{
  *first = 13;
  return first;
}

void TestSearchDispatchImplicit()
{
  thrust::device_vector<int> vec(10);
  thrust::device_vector<int> pattern(2);

  thrust::search(thrust::retag<my_tag>(vec.begin()),
                 thrust::retag<my_tag>(vec.end()),
                 thrust::retag<my_tag>(pattern.begin()),
                 thrust::retag<my_tag>(pattern.end()));

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestSearchDispatchImplicit);

template <typename T>
struct TestSearch
{
  void operator()(const size_t n)
  {
    if (n < 10)
    {
      return;
    }

    thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
    thrust::device_vector<T> d_data = h_data;

    // Create a pattern from a slice of the data
    size_t pattern_size = n / 10;
    size_t pattern_pos  = n / 3;

    thrust::host_vector<T> h_pattern(h_data.begin() + pattern_pos, h_data.begin() + pattern_pos + pattern_size);
    thrust::device_vector<T> d_pattern = h_pattern;

    typename thrust::host_vector<T>::iterator h_iter;
    typename thrust::device_vector<T>::iterator d_iter;

    h_iter = thrust::search(h_data.begin(), h_data.end(), h_pattern.begin(), h_pattern.end());
    d_iter = thrust::search(d_data.begin(), d_data.end(), d_pattern.begin(), d_pattern.end());
    ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());

    // Test with a pattern that doesn't exist
    thrust::host_vector<T> h_nonexistent(pattern_size, T(-1));
    thrust::device_vector<T> d_nonexistent = h_nonexistent;

    h_iter = thrust::search(h_data.begin(), h_data.end(), h_nonexistent.begin(), h_nonexistent.end());
    d_iter = thrust::search(d_data.begin(), d_data.end(), d_nonexistent.begin(), d_nonexistent.end());
    ASSERT_EQUAL(h_iter - h_data.begin(), d_iter - d_data.begin());
    ASSERT_EQUAL(h_iter - h_data.begin(), static_cast<typename thrust::host_vector<T>::difference_type>(h_data.size()));
    ASSERT_EQUAL(d_iter - d_data.begin(),
                 static_cast<typename thrust::device_vector<T>::difference_type>(d_data.size()));
  }
};
VariableUnitTest<TestSearch, SignedIntegralTypes> TestSearchInstance;

void TestSearchWithBigIndexesHelper(int magnitude)
{
  thrust::counting_iterator<long long> begin(0);
  thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  ASSERT_EQUAL(::cuda::std::distance(begin, end), 1ll << magnitude);

  // Create a pattern {17, 18}
  thrust::host_vector<long long> pattern{17, 18};
  thrust::device_vector<long long> d_pattern = pattern;

  // This should find the pattern at position 17
  auto result = thrust::search(thrust::device, begin, end, d_pattern.begin(), d_pattern.end());

  cuda::std::intmax_t distance = ::cuda::std::distance(begin, result);
  ASSERT_EQUAL(distance, 17);
}

void TestSearchWithBigIndexes()
{
  TestSearchWithBigIndexesHelper(20);
  TestSearchWithBigIndexesHelper(30);
}
DECLARE_UNITTEST(TestSearchWithBigIndexes);

// Test single element pattern
template <class Vector>
void TestSearchSingleElement()
{
  Vector data{0, 1, 2, 3, 4, 5};
  Vector pattern{3};

  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end());
  ASSERT_EQUAL(iter - data.begin(), 3);

  // Single element not found
  Vector pattern2{9};
  iter = thrust::search(data.begin(), data.end(), pattern2.begin(), pattern2.end());
  ASSERT_EQUAL(iter - data.begin(), static_cast<typename Vector::difference_type>(data.size()));
}
DECLARE_VECTOR_UNITTEST(TestSearchSingleElement);

// Test pattern equals entire data
template <class Vector>
void TestSearchFullMatch()
{
  Vector data{1, 2, 3, 4, 5};
  Vector pattern{1, 2, 3, 4, 5};

  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end());
  ASSERT_EQUAL(iter - data.begin(), 0); // Should find at beginning
}
DECLARE_VECTOR_UNITTEST(TestSearchFullMatch);

// Test overlapping patterns
template <class Vector>
void TestSearchOverlapping()
{
  Vector data{1, 1, 1, 2, 1, 1, 2};
  Vector pattern{1, 1, 2};

  // Should find first occurrence at position 1
  auto iter = thrust::search(data.begin(), data.end(), pattern.begin(), pattern.end());
  ASSERT_EQUAL(iter - data.begin(), 1);
}
DECLARE_VECTOR_UNITTEST(TestSearchOverlapping);
