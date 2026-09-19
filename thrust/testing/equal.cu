#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/equal.h>
#include <thrust/functional.h>
#include <thrust/iterator/retag.h>

#include <unittest/unittest.h>

template <class Vector>
void TestEqualSimple()
{
  using T = typename Vector::value_type;

  Vector v1{5, 2, 0, 0, 0};
  Vector v2{5, 2, 0, 6, 1};

  REQUIRE(thrust::equal(v1.begin(), v1.end(), v1.begin()));
  REQUIRE_FALSE(thrust::equal(v1.begin(), v1.end(), v2.begin()));
  REQUIRE(thrust::equal(v2.begin(), v2.end(), v2.begin()));

  REQUIRE(thrust::equal(v1.begin(), v1.begin() + 0, v1.begin()));
  REQUIRE(thrust::equal(v1.begin(), v1.begin() + 1, v1.begin()));
  REQUIRE(thrust::equal(v1.begin(), v1.begin() + 3, v2.begin()));
  REQUIRE_FALSE(thrust::equal(v1.begin(), v1.begin() + 4, v2.begin()));

  REQUIRE(thrust::equal(v1.begin(), v1.end(), v2.begin(), ::cuda::std::less_equal<T>()));
  REQUIRE_FALSE(thrust::equal(v1.begin(), v1.end(), v2.begin(), ::cuda::std::greater<T>()));
}
DECLARE_VECTOR_UNITTEST(TestEqualSimple);

template <typename T>
void TestEqual(const size_t n)
{
  thrust::host_vector<T> h_data1   = unittest::random_samples<T>(n);
  thrust::host_vector<T> h_data2   = unittest::random_samples<T>(n);
  thrust::device_vector<T> d_data1 = h_data1;
  thrust::device_vector<T> d_data2 = h_data2;

  // empty ranges
  REQUIRE(thrust::equal(h_data1.begin(), h_data1.begin(), h_data1.begin()));
  REQUIRE(thrust::equal(d_data1.begin(), d_data1.begin(), d_data1.begin()));

  // symmetric cases
  REQUIRE(thrust::equal(h_data1.begin(), h_data1.end(), h_data1.begin()));
  REQUIRE(thrust::equal(d_data1.begin(), d_data1.end(), d_data1.begin()));

  if (n > 0)
  {
    h_data1[0] = 0;
    h_data2[0] = 1;
    d_data1[0] = 0;
    d_data2[0] = 1;

    // different vectors
    REQUIRE_FALSE(thrust::equal(h_data1.begin(), h_data1.end(), h_data2.begin()));
    REQUIRE_FALSE(thrust::equal(d_data1.begin(), d_data1.end(), d_data2.begin()));

    // different predicates
    REQUIRE(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), ::cuda::std::less<T>()));
    REQUIRE(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), ::cuda::std::less<T>()));
    REQUIRE_FALSE(thrust::equal(h_data1.begin(), h_data1.begin() + 1, h_data2.begin(), ::cuda::std::greater<T>()));
    REQUIRE_FALSE(thrust::equal(d_data1.begin(), d_data1.begin() + 1, d_data2.begin(), ::cuda::std::greater<T>()));
  }
}
DECLARE_VARIABLE_UNITTEST(TestEqual);

template <typename InputIterator1, typename InputIterator2>
bool equal(my_system& system, InputIterator1 /*first*/, InputIterator1, InputIterator2)
{
  system.validate_dispatch();
  return false;
}

void TestEqualDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::equal(sys, vec.begin(), vec.end(), vec.begin());

  REQUIRE(sys.is_valid());
}
DECLARE_UNITTEST(TestEqualDispatchExplicit);

template <typename InputIterator1, typename InputIterator2>
bool equal(my_tag, InputIterator1 first, InputIterator1, InputIterator2)
{
  *first = 13;
  return false;
}

void TestEqualDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::equal(
    thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::retag<my_tag>(vec.begin()));

  REQUIRE(13 == vec.front());
}
DECLARE_UNITTEST(TestEqualDispatchImplicit);

struct only_set_when_both_expected
{
  long long expected;
  bool* flag;

  _CCCL_DEVICE bool operator()(long long x, long long y)
  {
    if (x == expected && y == expected)
    {
      *flag = true;
    }

    return x == y;
  }
};

void TestEqualWithBigIndexesHelper(int magnitude)
{
  const thrust::counting_iterator<long long> begin(1);
  const thrust::counting_iterator<long long> end = begin + (1ll << magnitude);
  REQUIRE(::cuda::std::distance(begin, end) == (1ll << magnitude));

  const thrust::device_ptr<bool> has_executed = thrust::device_malloc<bool>(1);
  *has_executed                               = false;

  const only_set_when_both_expected fn = {(1ll << magnitude) - 1, thrust::raw_pointer_cast(has_executed)};

  REQUIRE(thrust::equal(thrust::device, begin, end, begin, fn));

  const bool has_executed_h = *has_executed;
  thrust::device_free(has_executed);

  REQUIRE(has_executed_h);
}

#ifndef THRUST_FORCE_32_BIT_OFFSET_TYPE
void TestEqualWithBigIndexes()
{
  TestEqualWithBigIndexesHelper(30);
  TestEqualWithBigIndexesHelper(31);
  TestEqualWithBigIndexesHelper(32);
  TestEqualWithBigIndexesHelper(33);
}
DECLARE_UNITTEST(TestEqualWithBigIndexes);
#endif // THRUST_FORCE_32_BIT_OFFSET_TYPE
