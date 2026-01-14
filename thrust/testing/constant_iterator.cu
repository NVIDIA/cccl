#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <unittest/unittest.h>

// ensure that we properly support thrust::constant_iterator from cuda::std
void TestConstantIteratorTraits()
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
DECLARE_UNITTEST(TestConstantIteratorTraits);

void TestConstantIteratorConstructFromConvertibleSystem()
{
  thrust::constant_iterator<int> default_system(13);

  thrust::constant_iterator<int, thrust::use_default, thrust::host_system_tag> host_system = default_system;
  ASSERT_EQUAL(*default_system, *host_system);

  thrust::constant_iterator<int, thrust::use_default, thrust::device_system_tag> device_system = default_system;
  ASSERT_EQUAL(*default_system, *device_system);
}
DECLARE_UNITTEST(TestConstantIteratorConstructFromConvertibleSystem);

void TestConstantIteratorIncrement()
{
  thrust::constant_iterator<int> lhs(0, 0);
  thrust::constant_iterator<int> rhs(0, 0);

  ASSERT_EQUAL(0, lhs - rhs);

  lhs++;

  ASSERT_EQUAL(1, lhs - rhs);

  lhs++;
  lhs++;

  ASSERT_EQUAL(3, lhs - rhs);

  lhs += 5;

  ASSERT_EQUAL(8, lhs - rhs);

  lhs -= 10;

  ASSERT_EQUAL(-2, lhs - rhs);
}
DECLARE_UNITTEST(TestConstantIteratorIncrement);
static_assert(cuda::std::is_trivially_copy_constructible<thrust::constant_iterator<int>>::value, "");
static_assert(cuda::std::is_trivially_copyable<thrust::constant_iterator<int>>::value, "");

void TestConstantIteratorIncrementBig()
{
  long long int n = 10000000000ULL;

  thrust::constant_iterator<long long int> begin(1);
  thrust::constant_iterator<long long int> end = begin + n;

  ASSERT_EQUAL(cuda::std::distance(begin, end), n);
}
DECLARE_UNITTEST(TestConstantIteratorIncrementBig);

void TestConstantIteratorComparison()
{
  thrust::constant_iterator<int> iter1(0);
  thrust::constant_iterator<int> iter2(0);

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);

  iter1++;

  ASSERT_EQUAL(1, iter1 - iter2);
  ASSERT_EQUAL(false, iter1 == iter2);

  iter2++;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);

  iter1 += 100;
  iter2 += 100;

  ASSERT_EQUAL(0, iter1 - iter2);
  ASSERT_EQUAL(true, iter1 == iter2);
}
DECLARE_UNITTEST(TestConstantIteratorComparison);

void TestMakeConstantIterator()
{
  // test one argument version
  thrust::constant_iterator<int> iter0 = thrust::make_constant_iterator<int>(13);

  ASSERT_EQUAL(13, *iter0);

  // test two argument version
  thrust::constant_iterator<int, cuda::std::intmax_t> iter1 =
    thrust::make_constant_iterator<int, cuda::std::intmax_t>(13, 7);

  ASSERT_EQUAL(13, *iter1);
  ASSERT_EQUAL(7, iter1 - iter0);

  // ensure CTAD words
  thrust::constant_iterator deduced_iter{42};
  static_assert(cuda::std::is_same_v<decltype(deduced_iter), thrust::constant_iterator<int>>);
  ASSERT_EQUAL(42, *deduced_iter);
}
DECLARE_UNITTEST(TestMakeConstantIterator);

template <typename Vector>
void TestConstantIteratorCopy()
{
  using ValueType = typename Vector::value_type;
  using ConstIter = thrust::constant_iterator<ValueType>;

  Vector result(4);

  ConstIter first = thrust::make_constant_iterator<ValueType>(7);
  ConstIter last  = first + result.size();
  thrust::copy(first, last, result.begin());

  Vector ref(4, 7);
  ASSERT_EQUAL(ref, result);
};
DECLARE_VECTOR_UNITTEST(TestConstantIteratorCopy);

template <typename Vector>
void TestConstantIteratorTransform()
{
  using T         = typename Vector::value_type;
  using ConstIter = thrust::constant_iterator<T>;

  Vector result(4);

  ConstIter first1 = thrust::make_constant_iterator<T>(7);
  ConstIter last1  = first1 + result.size();
  ConstIter first2 = thrust::make_constant_iterator<T>(3);

  thrust::transform(first1, last1, result.begin(), cuda::std::negate<T>());

  Vector ref(4, -7);
  ASSERT_EQUAL(ref, result);

  thrust::transform(first1, last1, first2, result.begin(), cuda::std::plus<T>());

  ref = Vector(4, 10);
  ASSERT_EQUAL(ref, result);
};
DECLARE_VECTOR_UNITTEST(TestConstantIteratorTransform);

void TestConstantIteratorReduce()
{
  using T         = int;
  using ConstIter = thrust::constant_iterator<T>;

  ConstIter first = thrust::make_constant_iterator<T>(7);
  ConstIter last  = first + 4;

  T sum = thrust::reduce(first, last);

  ASSERT_EQUAL(sum, 4 * 7);
};
DECLARE_UNITTEST(TestConstantIteratorReduce);
