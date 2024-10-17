#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

#include <unittest/unittest.h>

template <typename Iterator>
class strided_range
{
public:
  using difference_type = typename thrust::iterator_difference<Iterator>::type;

  struct stride_functor
  {
    difference_type stride;

    stride_functor(difference_type stride)
        : stride(stride)
    {}

    _CCCL_HOST_DEVICE difference_type operator()(const difference_type& i) const
    {
      return stride * i;
    }
  };

  using CountingIterator    = typename thrust::counting_iterator<difference_type>;
  using TransformIterator   = typename thrust::transform_iterator<stride_functor, CountingIterator>;
  using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

  // type of the strided_range iterator
  using iterator = PermutationIterator;

  // construct strided_range for the range [first,last)
  strided_range(Iterator first, Iterator last, difference_type stride)
      : first(first)
      , last(last)
      , stride(stride)
  {}

  iterator begin() const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
  }

  iterator end() const
  {
    return begin() + ((last - first) + (stride - 1)) / stride;
  }

protected:
  Iterator first;
  Iterator last;
  difference_type stride;
};

template <class Vector>
void TestSortPermutationIterator()
{
  using Iterator = typename Vector::iterator;

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::sort(S.begin(), S.end());

  Vector ref{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQUAL(A, ref);
}
DECLARE_VECTOR_UNITTEST(TestSortPermutationIterator);

template <class Vector>
void TestStableSortPermutationIterator()
{
  using Iterator = typename Vector::iterator;

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};

  strided_range<Iterator> S(A.begin(), A.end(), 2);

  thrust::stable_sort(S.begin(), S.end());

  Vector ref{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQUAL(A, ref);
}
DECLARE_VECTOR_UNITTEST(TestStableSortPermutationIterator);

template <class Vector>
void TestSortByKeyPermutationIterator()
{
  using Iterator = typename Vector::iterator;

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};
  Vector B{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::sort_by_key(S.begin(), S.end(), T.begin());

  Vector ref_A{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQUAL(A, ref_A);

  Vector ref_B{2, 1, 0, 3, 4, 5, 8, 7, 6, 9};
  ASSERT_EQUAL(B, ref_B);
}
DECLARE_VECTOR_UNITTEST(TestSortByKeyPermutationIterator);

template <class Vector>
void TestStableSortByKeyPermutationIterator()
{
  using Iterator = typename Vector::iterator;

  Vector A{2, 9, 0, 1, 5, 3, 8, 6, 7, 4};
  Vector B{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  strided_range<Iterator> S(A.begin(), A.end(), 2);
  strided_range<Iterator> T(B.begin(), B.end(), 2);

  thrust::stable_sort_by_key(S.begin(), S.end(), T.begin());

  Vector ref_A{0, 9, 2, 1, 5, 3, 7, 6, 8, 4};
  ASSERT_EQUAL(A, ref_A);

  Vector ref_B{2, 1, 0, 3, 4, 5, 8, 7, 6, 9};
  ASSERT_EQUAL(B, ref_B);
}
DECLARE_VECTOR_UNITTEST(TestStableSortByKeyPermutationIterator);
