#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

// this example illustrates how to make repeated access to a range of values
// examples:
//   repeated_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   repeated_range([0, 1, 2, 3], 2) -> [0, 0, 1, 1, 2, 2, 3, 3]
//   repeated_range([0, 1, 2, 3], 3) -> [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
//   ...

template <typename Iterator>
class repeated_range
{
public:
  using difference_type = typename cuda::std::iterator_traits<Iterator>::difference_type;

  struct repeat_functor
  {
    difference_type repeats;

    repeat_functor(difference_type repeats)
        : repeats(repeats)
    {}

    __host__ __device__ difference_type operator()(const difference_type& i) const
    {
      return i / repeats;
    }
  };

  using CountingIterator    = typename thrust::counting_iterator<difference_type>;
  using TransformIterator   = typename thrust::transform_iterator<repeat_functor, CountingIterator>;
  using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

  // type of the repeated_range iterator
  using iterator = PermutationIterator;

  // construct repeated_range for the range [first,last)
  repeated_range(Iterator first, Iterator last, difference_type repeats)
      : first(first)
      , last(last)
      , repeats(repeats)
  {}

  iterator begin() const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), repeat_functor(repeats)));
  }

  iterator end() const
  {
    return begin() + repeats * (last - first);
  }

protected:
  Iterator first;
  Iterator last;
  difference_type repeats;
};

int main()
{
  thrust::device_vector<int> data{10, 20, 30, 40};

  // print the initial data
  std::cout << "range        ";
  thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  using Iterator = thrust::device_vector<int>::iterator;

  // create repeated_range with elements repeated twice
  repeated_range<Iterator> twice(data.begin(), data.end(), 2);
  std::cout << "repeated x2: ";
  thrust::copy(twice.begin(), twice.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // create repeated_range with elements repeated x3
  repeated_range<Iterator> thrice(data.begin(), data.end(), 3);
  std::cout << "repeated x3: ";
  thrust::copy(thrice.begin(), thrice.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
