#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <unittest/unittest.h>

// ensure that we properly support thrust::constant_iterator from cuda::std
TEST_CASE("TestConstantIteratorTraits", "[constant_iterator]")
{
  using it       = thrust::constant_iterator<int>;
  using traits   = cuda::std::iterator_traits<it>;
  using category = thrust::detail::iterator_category_with_system_and_traversal<::cuda::std::random_access_iterator_tag,
                                                                               thrust::any_system_tag,
                                                                               thrust::random_access_traversal_tag>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, signed int>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, category>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__has_random_access_traversal<it>);

  static_assert(!cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}

TEST_CASE("TestConstantIteratorConstructFromConvertibleSystem", "[constant_iterator]")
{
  const thrust::constant_iterator<int> default_system(13);

  const thrust::constant_iterator<int, thrust::use_default, thrust::host_system_tag> host_system = default_system;
  REQUIRE(*default_system == *host_system);

  const thrust::constant_iterator<int, thrust::use_default, thrust::device_system_tag> device_system = default_system;
  REQUIRE(*default_system == *device_system);
}

TEST_CASE("TestConstantIteratorIncrement", "[constant_iterator]")
{
  thrust::constant_iterator<int> lhs(0, 0);
  const thrust::constant_iterator<int> rhs(0, 0);

  REQUIRE(0 == lhs - rhs);

  lhs++;

  REQUIRE(1 == lhs - rhs);

  lhs++;
  lhs++;

  REQUIRE(3 == lhs - rhs);

  lhs += 5;

  REQUIRE(8 == lhs - rhs);

  lhs -= 10;

  REQUIRE(-2 == lhs - rhs);
}
static_assert(cuda::std::is_trivially_copy_constructible<thrust::constant_iterator<int>>::value);
static_assert(cuda::std::is_trivially_copyable<thrust::constant_iterator<int>>::value);

TEST_CASE("TestConstantIteratorIncrementBig", "[constant_iterator]")
{
  const long long int n = 10000000000ULL;

  const thrust::constant_iterator<long long int> begin(1);
  const thrust::constant_iterator<long long int> end = begin + n;

  REQUIRE(cuda::std::distance(begin, end) == n);
}

TEST_CASE("TestConstantIteratorComparison", "[constant_iterator]")
{
  thrust::constant_iterator<int> iter1(0);
  thrust::constant_iterator<int> iter2(0);

  REQUIRE(0 == iter1 - iter2);
  REQUIRE(iter1 == iter2);

  iter1++;

  REQUIRE(1 == iter1 - iter2);
  REQUIRE_FALSE(iter1 == iter2);

  iter2++;

  REQUIRE(0 == iter1 - iter2);
  REQUIRE(iter1 == iter2);

  iter1 += 100;
  iter2 += 100;

  REQUIRE(0 == iter1 - iter2);
  REQUIRE(iter1 == iter2);
}

TEST_CASE("TestMakeConstantIterator", "[constant_iterator]")
{
  // test one argument version
  const thrust::constant_iterator<int> iter0 = thrust::make_constant_iterator<int>(13);

  REQUIRE(13 == *iter0);

  // test two argument version
  const thrust::constant_iterator<int, cuda::std::intmax_t> iter1 =
    thrust::make_constant_iterator<int, cuda::std::intmax_t>(13, 7);

  REQUIRE(13 == *iter1);
  REQUIRE(7 == iter1 - iter0);

  // ensure CTAD words
  // NOLINTNEXTLINE(misc-const-correctness): decltype must not be const-qualified
  thrust::constant_iterator deduced_iter{42};
  static_assert(cuda::std::is_same_v<decltype(deduced_iter), thrust::constant_iterator<int>>);
  REQUIRE(42 == *deduced_iter);
}

template <typename Vector>
void test_constant_iterator_copy()
{
  using ValueType = typename Vector::value_type;
  using ConstIter = thrust::constant_iterator<ValueType>;

  Vector result(4);

  const ConstIter first = thrust::make_constant_iterator<ValueType>(7);
  const ConstIter last  = first + result.size();
  thrust::copy(first, last, result.begin());

  Vector ref(4, 7);
  REQUIRE(ref == result);
};
DECLARE_VECTOR_UNITTEST(test_constant_iterator_copy);

template <typename Vector>
void test_constant_iterator_transform()
{
  using T         = typename Vector::value_type;
  using ConstIter = thrust::constant_iterator<T>;

  Vector result(4);

  const ConstIter first1 = thrust::make_constant_iterator<T>(7);
  const ConstIter last1  = first1 + result.size();
  const ConstIter first2 = thrust::make_constant_iterator<T>(3);

  thrust::transform(first1, last1, result.begin(), cuda::std::negate<T>());

  Vector ref(4, -7);
  REQUIRE(ref == result);

  thrust::transform(first1, last1, first2, result.begin(), cuda::std::plus<T>());

  ref = Vector(4, 10);
  REQUIRE(ref == result);
};
DECLARE_VECTOR_UNITTEST(test_constant_iterator_transform);

TEST_CASE("TestConstantIteratorReduce", "[constant_iterator]")
{
  using T         = int;
  using ConstIter = thrust::constant_iterator<T>;

  const ConstIter first = thrust::make_constant_iterator<T>(7);
  const ConstIter last  = first + 4;

  const T sum = thrust::reduce(first, last);

  REQUIRE(sum == 4 * 7);
}
