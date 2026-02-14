#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/tabulate.h>
#include <thrust/universal_vector.h>

#include <cuda/iterator>
#include <cuda/std/mdspan>

#include <iomanip>
#include <iostream>

#include "include/host_device.h"

int main()
{
  const int rows = 32;
  const int cols = 16;

  // Create a 2D multidimensional array of ints.
  thrust::universal_vector<int> data(rows * cols, 42);
  cuda::std::mdspan M(thrust::raw_pointer_cast(data.data()), rows, cols);

  // Create an iterator to the flat linear index space for the multidimensional array.
  auto flat_idx = cuda::counting_iterator(0);

  // Fill the array with pseudorandom inputs in parallel on the device.
  thrust::tabulate(thrust::device, M.data_handle(), M.data_handle() + M.size(), [] __host__ __device__(int flat) {
    thrust::default_random_engine rng;
    thrust::uniform_int_distribution<int> dist(0, 3);
    rng.discard(flat); // Advance to the current element's position.
    return dist(rng);
  });

  // Create a range to the column index of each element.
  auto col_idx_begin = thrust::make_transform_iterator(flat_idx, [=] __host__ __device__(int flat) {
    return flat / rows;
  });
  auto col_idx_end   = col_idx_begin + M.size();

  // Create a transposed view of the multidimensional array.
  auto M_transposed = thrust::make_permutation_iterator(
    M.data_handle(), thrust::make_transform_iterator(cuda::counting_iterator(0), [=] __host__ __device__(int flat) {
      int i = flat / cols;
      int j = flat % cols;
      return i + j * rows;
    }));

  // Sum each column, storing the result in a new vector.
  thrust::universal_vector<int> sums(cols);
  thrust::reduce_by_key(
    thrust::device, col_idx_begin, col_idx_end, M_transposed, thrust::make_discard_iterator(), sums.begin());

  // Output the result.
  thrust::for_each_n(thrust::seq, flat_idx, rows, [&](int i) {
    std::cout << "[ ";
    thrust::for_each_n(thrust::seq, flat_idx, cols, [&](int j) {
      std::cout << std::setw(2) << M(i, j) << " ";
    });
    std::cout << "]\n";
  });

  std::cout << "  ";
  thrust::for_each_n(thrust::seq, flat_idx, cols, [&](int) {
    std::cout << " = ";
  });
  std::cout << "\n";

  std::cout << "  ";
  thrust::for_each_n(thrust::seq, flat_idx, cols, [&](int j) {
    std::cout << std::setw(2) << sums[j] << " ";
  });
  std::cout << "\n";
}
