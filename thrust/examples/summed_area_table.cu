#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <iomanip>
#include <iostream>

// This example computes a summed area table using segmented scan
// http://en.wikipedia.org/wiki/Summed_area_table

// transpose an M-by-N array
template <typename T>
void transpose(size_t m, size_t n, thrust::device_vector<T>& src, thrust::device_vector<T>& dst)
{
  thrust::counting_iterator<size_t> indices(0);

  // lambda to convert a linear index to a linear index in the transpose
  auto transpose_index = [m, n] __device__(size_t linear_index) {
    size_t i = linear_index / n;
    size_t j = linear_index % n;
    return m * j + i;
  };

  thrust::gather(thrust::make_transform_iterator(indices, transpose_index),
                 thrust::make_transform_iterator(indices, transpose_index) + dst.size(),
                 src.begin(),
                 dst.begin());
}

// scan the rows of an M-by-N array
template <typename T>
void scan_horizontally(size_t n, thrust::device_vector<T>& d_data)
{
  thrust::counting_iterator<size_t> indices(0);

  // lambda to convert a linear index to a row index
  auto row_index = [n] __device__(size_t i) {
    return i / n;
  };

  thrust::inclusive_scan_by_key(
    thrust::make_transform_iterator(indices, row_index),
    thrust::make_transform_iterator(indices, row_index) + d_data.size(),
    d_data.begin(),
    d_data.begin());
}

// print an M-by-N array
template <typename T>
void print(size_t m, size_t n, thrust::device_vector<T>& d_data)
{
  thrust::host_vector<T> h_data = d_data;

  for (size_t i = 0; i < m; i++)
  {
    for (size_t j = 0; j < n; j++)
    {
      std::cout << std::setw(8) << h_data[i * n + j] << " ";
    }
    std::cout << "\n";
  }
}

int main()
{
  size_t m = 3; // number of rows
  size_t n = 4; // number of columns

  // 2d array stored in row-major order [(0,0), (0,1), (0,2) ... ]
  thrust::device_vector<int> data(m * n, 1);

  std::cout << "[step 0] initial array" << std::endl;
  print(m, n, data);

  std::cout << "[step 1] scan horizontally" << std::endl;
  scan_horizontally(n, data);
  print(m, n, data);

  std::cout << "[step 2] transpose array" << std::endl;
  thrust::device_vector<int> temp(m * n);
  transpose(m, n, data, temp);
  print(n, m, temp);

  std::cout << "[step 3] scan transpose horizontally" << std::endl;
  scan_horizontally(m, temp);
  print(n, m, temp);

  std::cout << "[step 4] transpose the transpose" << std::endl;
  transpose(n, m, temp, data);
  print(m, n, data);

  return 0;
}
