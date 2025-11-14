#include <thrust/distance.h>
#include <thrust/iterator/offset_iterator.h>

#include <cuda/std/iterator>

#include <unittest/unittest.h>

// ensure that we properly support thrust::counting_iterator from cuda::std
void TestOffsetIteratorTraits()
{
  using base_it    = thrust::host_vector<int>::iterator;
  using it         = thrust::offset_iterator<base_it>;
  using traits     = cuda::std::iterator_traits<it>;
  using vec_traits = cuda::std::iterator_traits<base_it>;

  static_assert(cuda::std::is_same_v<traits::difference_type, vec_traits::difference_type>);
  static_assert(cuda::std::is_same_v<traits::value_type, vec_traits::value_type>);
  static_assert(cuda::std::is_same_v<traits::pointer, vec_traits::pointer>);
  static_assert(cuda::std::is_same_v<traits::reference, vec_traits::reference>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, vec_traits::iterator_category>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__has_random_access_traversal<it>);

  static_assert(cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}
DECLARE_UNITTEST(TestOffsetIteratorTraits);

template <typename Vector>
void TestOffsetConstructor()
{
  thrust::offset_iterator<int*> iter0;
  ASSERT_EQUAL(iter0.base(), static_cast<int*>(nullptr));
  ASSERT_EQUAL(iter0.offset(), 0);

  Vector v{42, 43};
  thrust::offset_iterator iter1(v.begin());
  ASSERT_EQUAL_QUIET(iter1.base(), v.begin());
  ASSERT_EQUAL(iter1.offset(), 0);
  ASSERT_EQUAL(*iter1, 42);

  thrust::offset_iterator iter2(v.begin(), 1);
  ASSERT_EQUAL_QUIET(iter2.base(), v.begin());
  ASSERT_EQUAL(iter2.offset(), 1);
  ASSERT_EQUAL(*iter2, 43);

  ptrdiff_t offset = 1;
  thrust::offset_iterator iter3(v.begin(), &offset);
  ASSERT_EQUAL_QUIET(iter3.base(), v.begin());
  ASSERT_EQUAL(iter3.offset(), &offset);
  ASSERT_EQUAL(*iter3.offset(), 1);
  ASSERT_EQUAL(*iter3, 43);
}
DECLARE_VECTOR_UNITTEST(TestOffsetConstructor);

template <typename Vector>
void TestOffsetIteratorCopyConstructorAndAssignment()
{
  Vector v{42, 43};

  // value offset
  {
    thrust::offset_iterator iter0(v.begin());
#if _CCCL_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // _CCCL_COMPILER(MSVC)
    thrust::offset_iterator iter1(iter0);
#endif // _CCCL_COMPILER(MSVC)
    ASSERT_EQUAL(iter0 == iter1, true);
    ASSERT_EQUAL(*iter0 == *iter1, true);

    thrust::offset_iterator iter2(v.begin() + 1);
    ASSERT_EQUAL(iter0 != iter2, true);
    ASSERT_EQUAL(*iter0 != *iter2, true);

    iter2 = iter0;
    ASSERT_EQUAL(iter0 == iter2, true);
    ASSERT_EQUAL(*iter0 == *iter2, true);
  }

  // indirect offset
  {
    const typename Vector::iterator::difference_type offset = 0;
    thrust::offset_iterator iter0(v.begin(), &offset);

#if _CCCL_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // _CCCL_COMPILER(MSVC)
    thrust::offset_iterator iter1(iter0);
#endif // _CCCL_COMPILER(MSVC)
    ASSERT_EQUAL(iter0 == iter1, true);
    ASSERT_EQUAL(*iter0 == *iter1, true);

    thrust::offset_iterator iter2(v.begin() + 1, &offset);
    ASSERT_EQUAL(iter0 != iter2, true);
    ASSERT_EQUAL(*iter0 != *iter2, true);

    iter2 = iter0;
    ASSERT_EQUAL(iter0 == iter2, true);
    ASSERT_EQUAL(*iter0 == *iter2, true);
  }
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorCopyConstructorAndAssignment);

template <typename Vector>
void TestOffsetIteratorIncrement()
{
  auto test = [](auto iter) {
    ASSERT_EQUAL(*iter, 0);
    iter++;
    ASSERT_EQUAL(*iter, 1);
    iter++;
    iter++;
    ASSERT_EQUAL(*iter, 3);
    iter += 5;
    ASSERT_EQUAL(*iter, 8);
    iter -= 10;
    ASSERT_EQUAL(*iter, -2);
  };

  const Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
  test(thrust::offset_iterator(v.begin() + 1, 1));
  const typename Vector::iterator::difference_type offset = 1;
  test(thrust::offset_iterator(v.begin() + 1, &offset));
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorIncrement);

template <typename Vector>
void TestOffsetIteratorMutation()
{
  {
    Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    thrust::offset_iterator it(v.begin() + 1, 1);
    *it = 42;
    ++it;
    *it = 43;
    ++it.offset();
    *it = 44;
    ASSERT_EQUAL(v, (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
  }
  {
    Vector v{-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    typename Vector::iterator::difference_type offset = 1;
    thrust::offset_iterator it(v.begin() + 1, &offset);
    *it = 42;
    ++it;
    *it    = 43;
    offset = 2;
    *it    = 44;
    ASSERT_EQUAL(v, (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
  }
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorMutation);

template <typename Vector>
void TestOffsetIteratorComparisonAndDistance()
{
  auto test = [](auto iter1, auto iter2) {
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(::cuda::std::distance(iter1, iter2), 0);

    iter1++;
    ASSERT_EQUAL(iter1 == iter2, false);
    ASSERT_EQUAL(iter1 - iter2, 1);
    ASSERT_EQUAL(::cuda::std::distance(iter1, iter2), -1);

    iter2++;
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(::cuda::std::distance(iter1, iter2), 0);

    iter1 += 100;
    iter2 += 100;
    ASSERT_EQUAL(iter1 == iter2, true);
    ASSERT_EQUAL(iter1 - iter2, 0);
    ASSERT_EQUAL(::cuda::std::distance(iter1, iter2), 0);

    iter1 -= 5;
    ASSERT_EQUAL(iter1 == iter2, false);
    ASSERT_EQUAL(iter1 - iter2, -5);
    ASSERT_EQUAL(::cuda::std::distance(iter1, iter2), 5);
  };

  Vector v(101);
  test(thrust::offset_iterator(v.begin()), thrust::offset_iterator(v.begin()));
  const typename Vector::iterator::difference_type offset = 0;
  test(thrust::offset_iterator(v.begin(), &offset), thrust::offset_iterator(v.begin(), &offset));
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorComparisonAndDistance);

template <typename Vector>
void TestOffsetIteratorLateValue()
{
  typename Vector::difference_type offset;
  Vector v{0, 1, 2, 3, 4, 5, 6, 7, 8};
  thrust::offset_iterator iter(v.begin(), &offset);
  offset = 2; // we provide the offset value **after** constructing the iterator
  ASSERT_EQUAL(*iter, 2);
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorLateValue);

template <typename Vector>
void TestOffsetIteratorIndirectValueFancyIterator()
{
  using thrust::placeholders::_1;

  Vector v{0, 1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<typename Vector::difference_type> offsets{2};
  auto it = thrust::make_transform_iterator(offsets.begin(), _1 * 3);
  thrust::offset_iterator iter(v.begin(), it);
  ASSERT_EQUAL(*iter, 6);
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorIndirectValueFancyIterator);
