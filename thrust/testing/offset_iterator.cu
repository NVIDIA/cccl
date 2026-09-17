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
  REQUIRE(iter0.base() == static_cast<int*>(nullptr));
  REQUIRE(iter0.offset() == 0);

  Vector v{42, 43};
  thrust::offset_iterator iter1(v.begin());
  ASSERT_EQUAL_QUIET(iter1.base(), v.begin());
  REQUIRE(iter1.offset() == 0);
  REQUIRE(*iter1 == 42);

  thrust::offset_iterator iter2(v.begin(), 1);
  ASSERT_EQUAL_QUIET(iter2.base(), v.begin());
  REQUIRE(iter2.offset() == 1);
  REQUIRE(*iter2 == 43);

  ptrdiff_t offset = 1;
  thrust::offset_iterator iter3(v.begin(), &offset);
  ASSERT_EQUAL_QUIET(iter3.base(), v.begin());
  REQUIRE(iter3.offset() == &offset);
  REQUIRE(*iter3.offset() == 1);
  REQUIRE(*iter3 == 43);
}
DECLARE_VECTOR_UNITTEST(TestOffsetConstructor);

template <typename Vector>
void TestOffsetIteratorCopyConstructorAndAssignment()
{
  Vector v{42, 43};

  // value offset
  {
    const thrust::offset_iterator iter0(v.begin());
#if _CCCL_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // _CCCL_COMPILER(MSVC)
    const thrust::offset_iterator iter1(iter0);
#endif // _CCCL_COMPILER(MSVC)
    REQUIRE(iter0 == iter1);
    REQUIRE(*iter0 == *iter1);

    thrust::offset_iterator iter2(v.begin() + 1);
    REQUIRE(iter0 != iter2);
    REQUIRE(*iter0 != *iter2);

    iter2 = iter0;
    REQUIRE(iter0 == iter2);
    REQUIRE(*iter0 == *iter2);
  }

  // indirect offset
  {
    const typename Vector::iterator::difference_type offset = 0;
    const thrust::offset_iterator iter0(v.begin(), &offset);

#if _CCCL_COMPILER(MSVC) // MSVC cannot deduce the template arguments from the copy ctor
    decltype(iter0) iter1(iter0);
#else // _CCCL_COMPILER(MSVC)
    const thrust::offset_iterator iter1(iter0);
#endif // _CCCL_COMPILER(MSVC)
    REQUIRE(iter0 == iter1);
    REQUIRE(*iter0 == *iter1);

    thrust::offset_iterator iter2(v.begin() + 1, &offset);
    REQUIRE(iter0 != iter2);
    REQUIRE(*iter0 != *iter2);

    iter2 = iter0;
    REQUIRE(iter0 == iter2);
    REQUIRE(*iter0 == *iter2);
  }
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorCopyConstructorAndAssignment);

template <typename Vector>
void TestOffsetIteratorIncrement()
{
  auto test = [](auto iter) {
    REQUIRE(*iter == 0);
    iter++;
    REQUIRE(*iter == 1);
    iter++;
    iter++;
    REQUIRE(*iter == 3);
    iter += 5;
    REQUIRE(*iter == 8);
    iter -= 10;
    REQUIRE(*iter == -2);
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
    REQUIRE(v == (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
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
    REQUIRE(v == (Vector{-2, -1, 42, 43, 44, 3, 4, 5, 6, 7, 8}));
  }
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorMutation);

template <typename Vector>
void TestOffsetIteratorComparisonAndDistance()
{
  auto test = [](auto iter1, auto iter2) {
    REQUIRE(iter1 == iter2);
    REQUIRE(iter1 - iter2 == 0);
    REQUIRE(::cuda::std::distance(iter1, iter2) == 0);

    iter1++;
    REQUIRE_FALSE(iter1 == iter2);
    REQUIRE(iter1 - iter2 == 1);
    REQUIRE(::cuda::std::distance(iter1, iter2) == -1);

    iter2++;
    REQUIRE(iter1 == iter2);
    REQUIRE(iter1 - iter2 == 0);
    REQUIRE(::cuda::std::distance(iter1, iter2) == 0);

    iter1 += 100;
    iter2 += 100;
    REQUIRE(iter1 == iter2);
    REQUIRE(iter1 - iter2 == 0);
    REQUIRE(::cuda::std::distance(iter1, iter2) == 0);

    iter1 -= 5;
    REQUIRE_FALSE(iter1 == iter2);
    REQUIRE(iter1 - iter2 == -5);
    REQUIRE(::cuda::std::distance(iter1, iter2) == 5);
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
  const thrust::offset_iterator iter(v.begin(), &offset);
  offset = 2; // we provide the offset value **after** constructing the iterator
  REQUIRE(*iter == 2);
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorLateValue);

template <typename Vector>
void TestOffsetIteratorIndirectValueFancyIterator()
{
  using thrust::placeholders::_1;

  Vector v{0, 1, 2, 3, 4, 5, 6, 7, 8};
  thrust::device_vector<typename Vector::difference_type> offsets{2};
  auto it = thrust::make_transform_iterator(offsets.begin(), _1 * 3);
  const thrust::offset_iterator iter(v.begin(), it);
  REQUIRE(*iter == 6);
}
DECLARE_VECTOR_UNITTEST(TestOffsetIteratorIndirectValueFancyIterator);
