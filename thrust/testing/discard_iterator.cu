#include <thrust/iterator/discard_iterator.h>

#include <cuda/std/type_traits>

#include <unittest/unittest.h>

void TestDiscardIteratorIncrement()
{
  thrust::discard_iterator<> lhs(0);
  thrust::discard_iterator<> rhs(0);

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
DECLARE_UNITTEST(TestDiscardIteratorIncrement);
static_assert(cuda::std::is_trivially_copy_constructible<thrust::discard_iterator<>>::value, "");
static_assert(cuda::std::is_trivially_copyable<thrust::discard_iterator<>>::value, "");

void TestDiscardIteratorComparison()
{
  thrust::discard_iterator<> iter1(0);
  thrust::discard_iterator<> iter2(0);

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
DECLARE_UNITTEST(TestDiscardIteratorComparison);

void TestMakeDiscardIterator()
{
  thrust::discard_iterator<> iter0 = thrust::make_discard_iterator(13);

  *iter0 = 7;

  thrust::discard_iterator<> iter1 = thrust::make_discard_iterator(7);

  *iter1 = 13;

  ASSERT_EQUAL(6, iter0 - iter1);
}
DECLARE_UNITTEST(TestMakeDiscardIterator);

void TestZippedDiscardIterator()
{
  using namespace thrust;

  using IteratorTuple1 = tuple<discard_iterator<>>;
  using ZipIterator1   = zip_iterator<IteratorTuple1>;

  IteratorTuple1 t = thrust::make_tuple(thrust::make_discard_iterator());

  ZipIterator1 z_iter1_first = thrust::make_zip_iterator(t);
  ZipIterator1 z_iter1_last  = z_iter1_first + 10;
  for (; z_iter1_first != z_iter1_last; ++z_iter1_first)
  {
    ;
  }

  ASSERT_EQUAL(10, thrust::get<0>(z_iter1_first.get_iterator_tuple()) - thrust::make_discard_iterator());

  using IteratorTuple2 = tuple<int*, discard_iterator<>>;
  using ZipIterator2   = zip_iterator<IteratorTuple2>;

  ZipIterator2 z_iter_first = thrust::make_zip_iterator(thrust::make_tuple((int*) 0, thrust::make_discard_iterator()));
  ZipIterator2 z_iter_last  = z_iter_first + 10;

  for (; z_iter_first != z_iter_last; ++z_iter_first)
  {
    ;
  }

  ASSERT_EQUAL(10, thrust::get<1>(z_iter_first.get_iterator_tuple()) - thrust::make_discard_iterator());
}
DECLARE_UNITTEST(TestZippedDiscardIterator);
