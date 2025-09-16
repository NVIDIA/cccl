#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iterator>

int main()
{
  // this example computes indices for all the nonzero values in a sequence

  // sequence of zero and nonzero values
  thrust::device_vector<int> stencil{0, 1, 1, 0, 0, 1, 0, 1};

  // storage for the nonzero indices
  thrust::device_vector<int> indices(8);

  // counting iterators define a sequence [0, 8)
  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + 8;

  // compute indices of nonzero elements
  using IndexIterator = thrust::device_vector<int>::iterator;

  IndexIterator indices_end = thrust::copy_if(first, last, stencil.begin(), indices.begin(), ::cuda::std::identity{});
  // indices now contains [1,2,5,7]

  // print result
  std::cout << "found " << cuda::std::distance(indices.begin(), indices_end) << " nonzero values at indices:\n";
  thrust::copy(indices.begin(), indices_end, std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
