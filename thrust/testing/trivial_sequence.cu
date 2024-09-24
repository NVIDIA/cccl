#include <thrust/detail/trivial_sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <unittest/unittest.h>

template <typename Iterator>
void test(Iterator first, Iterator last)
{
  using System = typename thrust::iterator_system<Iterator>::type;
  System system;
  thrust::detail::trivial_sequence<Iterator, System> ts(system, first, last);
  using ValueType = typename thrust::iterator_traits<Iterator>::value_type;

  ASSERT_EQUAL_QUIET((ValueType) ts.begin()[0], ValueType(0, 11));
  ASSERT_EQUAL_QUIET((ValueType) ts.begin()[1], ValueType(2, 11));
  ASSERT_EQUAL_QUIET((ValueType) ts.begin()[2], ValueType(1, 13));
  ASSERT_EQUAL_QUIET((ValueType) ts.begin()[3], ValueType(0, 10));
  ASSERT_EQUAL_QUIET((ValueType) ts.begin()[4], ValueType(1, 12));

  ts.begin()[0] = ValueType(0, 0);
  ts.begin()[1] = ValueType(0, 0);
  ts.begin()[2] = ValueType(0, 0);
  ts.begin()[3] = ValueType(0, 0);
  ts.begin()[4] = ValueType(0, 0);

  using TrivialIterator = typename thrust::detail::trivial_sequence<Iterator, System>::iterator_type;

  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<Iterator>::value, false);
  ASSERT_EQUAL((bool) thrust::is_contiguous_iterator<TrivialIterator>::value, true);
}

template <class Vector>
void TestTrivialSequence()
{
  Vector A{0, 2, 1, 0, 1};
  Vector B{11, 11, 13, 10, 12};

  test(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin())),
       thrust::make_zip_iterator(thrust::make_tuple(A.end(), B.end())));

  Vector refA{0, 2, 1, 0, 1};
  ASSERT_EQUAL(A, refA);
  // ensure that values weren't modified
  Vector refB{11, 11, 13, 10, 12};
  ASSERT_EQUAL(B, refB);
}
DECLARE_VECTOR_UNITTEST(TestTrivialSequence);
