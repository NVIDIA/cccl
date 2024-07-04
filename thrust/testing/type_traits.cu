#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <unittest/unittest.h>

void TestIsContiguousIterator()
{
  using HostVector   = thrust::host_vector<int>;
  using DeviceVector = thrust::device_vector<int>;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<int*>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<HostVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::iterator>::value, true);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<DeviceVector::const_iterator>::value, true);

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<thrust::device_ptr<int>>::value, true);

  using HostIteratorTuple = thrust::tuple<HostVector::iterator, HostVector::iterator>;

  using ConstantIterator  = thrust::constant_iterator<int>;
  using CountingIterator  = thrust::counting_iterator<int>;
  using TransformIterator = thrust::transform_iterator<thrust::identity<int>, HostVector::iterator>;
  using ZipIterator       = thrust::zip_iterator<HostIteratorTuple>;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ConstantIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<CountingIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TransformIterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<ZipIterator>::value, false);
}
DECLARE_UNITTEST(TestIsContiguousIterator);

void TestIsCommutative()
{
  {
    using T  = int;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::multiplies<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::minimum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::maximum<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::logical_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::logical_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_or<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_and<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = int;
    using Op = thrust::bit_xor<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = char;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = short;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = long long;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = float;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }
  {
    using T  = double;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, true);
  }

  {
    using T  = int;
    using Op = thrust::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = int;
    using Op = thrust::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = thrust::divides<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
  {
    using T  = float;
    using Op = thrust::minus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }

  {
    using T  = thrust::tuple<int, int>;
    using Op = thrust::plus<T>;
    ASSERT_EQUAL((bool) thrust::detail::is_commutative<Op>::value, false);
  }
}
DECLARE_UNITTEST(TestIsCommutative);
