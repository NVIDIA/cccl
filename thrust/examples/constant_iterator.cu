#define CCCL_IGNORE_DEPRECATED_API

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cuda/iterator>

#include <iostream>
#include <iterator>

// TODO(jfaibussowit): Remove when CI clang-tidy is bumped to clang-22
int main() // NOLINT(bugprone-exception-escape)
{
  thrust::device_vector<int> data{3, 7, 2, 5};

  // add 10 to all values in data
  thrust::transform(data.begin(), data.end(), cuda::constant_iterator<int>(10), data.begin(), cuda::std::plus<int>());

  // data is now [13, 17, 12, 15]

  // print result
  thrust::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}
