#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>

#include <cstddef>
#include <iostream>
#include <iterator>

// This example decodes a run-length code [1] for an array of characters.
//
// [1] http://en.wikipedia.org/wiki/Run-length_encoding

int main()
{
  // compressed input and run lengths
  thrust::device_vector<char> input  = {'a', 'b', 'c', 'd', 'e', 'f'};
  thrust::device_vector<int> lengths = {3, 5, 1, 2, 9, 2};

  // print the initial data
  std::cout << "run-length encoded input:" << '\n';
  for (size_t i = 0; i < input.size(); i++)
  {
    std::cout << "(" << input[i] << "," << lengths[i] << ")";
  }
  std::cout << '\n' << '\n';

  // scan the lengths
  thrust::inclusive_scan(lengths.begin(), lengths.end(), lengths.begin());

  // output size is sum of the run lengths
  const int N = lengths.back();

  // compute input index for each output element
  thrust::device_vector<int> indices(N);
  thrust::lower_bound(
    lengths.begin(),
    lengths.end(),
    thrust::counting_iterator<int>(1),
    thrust::counting_iterator<int>(N + 1),
    indices.begin());

  // gather input elements
  thrust::device_vector<char> output(N);
  thrust::gather(indices.begin(), indices.end(), input.begin(), output.begin());

  // print the initial data
  std::cout << "decoded output:" << '\n';
  thrust::copy(output.begin(), output.end(), std::ostream_iterator<char>(std::cout, ""));
  std::cout << '\n';

  return 0;
}
