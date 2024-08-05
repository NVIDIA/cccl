#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <iostream>

#include "include/host_device.h"

// this example illustrates how to tile a range multiple times
// examples:
//   tiled_range([0, 1, 2, 3], 1) -> [0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 2) -> [0, 1, 2, 3, 0, 1, 2, 3]
//   tiled_range([0, 1, 2, 3], 3) -> [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
//   ...

template <typename Iterator>
class tiled_range
{
public:
  using difference_type = typename thrust::iterator_difference<Iterator>::type;

  struct tile_functor
  {
    difference_type tile_size;

    tile_functor(difference_type tile_size)
        : tile_size(tile_size)
    {}

    __host__ __device__ difference_type operator()(const difference_type& i) const
    {
      return i % tile_size;
    }
  };

  using CountingIterator    = typename thrust::counting_iterator<difference_type>;
  using TransformIterator   = typename thrust::transform_iterator<tile_functor, CountingIterator>;
  using PermutationIterator = typename thrust::permutation_iterator<Iterator, TransformIterator>;

  // type of the tiled_range iterator
  using iterator = PermutationIterator;

  // construct repeated_range for the range [first,last)
  tiled_range(Iterator first, Iterator last, difference_type tiles)
      : first(first)
      , last(last)
      , tiles(tiles)
  {}

  iterator begin() const
  {
    return PermutationIterator(first, TransformIterator(CountingIterator(0), tile_functor(last - first)));
  }

  iterator end() const
  {
    return begin() + tiles * (last - first);
  }

protected:
  Iterator first;
  Iterator last;
  difference_type tiles;
};

int main()
{
  thrust::device_vector<int> data(4);
  data[0] = 10;
  data[1] = 20;
  data[2] = 30;
  data[3] = 40;

  // print the initial data
  std::cout << "range        ";
  thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  using Iterator = thrust::device_vector<int>::iterator;

  // create tiled_range with two tiles
  tiled_range<Iterator> two(data.begin(), data.end(), 2);
  std::cout << "two tiles:   ";
  thrust::copy(two.begin(), two.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // create tiled_range with three tiles
  tiled_range<Iterator> three(data.begin(), data.end(), 3);
  std::cout << "three tiles: ";
  thrust::copy(three.begin(), three.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  return 0;
}
