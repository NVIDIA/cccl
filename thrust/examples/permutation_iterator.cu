#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>

#include <iostream>

// this example fuses a gather operation with a reduction for
// greater efficiency than separate gather() and reduce() calls

int main()
{
  // gather locations
  thrust::device_vector<int> map = {3, 1, 0, 5};

  // array to gather from
  thrust::device_vector<int> source = {10, 20, 30, 40, 50, 60};

  // fuse gather with reduction:
  //   sum = source[map[0]] + source[map[1]] + ...
  int sum = thrust::reduce(thrust::make_permutation_iterator(source.begin(), map.begin()),
                           thrust::make_permutation_iterator(source.begin(), map.end()));

  // print sum
  std::cout << "sum is " << sum << std::endl;

  return 0;
}
