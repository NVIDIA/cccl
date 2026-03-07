#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform_reduce.h>

// compute minimum and maximum values in a single reduction

// minmax_pair stores the minimum and maximum
// values that have been encountered so far
template <typename T>
struct minmax_pair
{
  T min_val;
  T max_val;
};

int main()
{
  // input size
  size_t N = 10;

  // initialize random number generator
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(10, 99);

  // initialize data on host
  thrust::host_vector<int> host_data(N);
  for (auto& e : host_data)
  {
    e = dist(rng);
  }
  thrust::device_vector<int> data = host_data;

  // lambda that takes in a value x and returns a minmax_pair whose minimum and maximum values are initialized to x
  auto unary_op = [] __device__(const int& x) {
    minmax_pair<int> result;
    result.min_val = x;
    result.max_val = x;
    return result;
  };

  // lambda that accepts two minmax_pair structs and returns a new minmax_pair whose minimum and
  // maximum values are the min() and max() respectively of the minimums and maximums of the input pairs
  auto binary_op = [] __device__(const minmax_pair<int>& x, const minmax_pair<int>& y) {
    minmax_pair<int> result;
    result.min_val = thrust::min(x.min_val, y.min_val);
    result.max_val = thrust::max(x.max_val, y.max_val);
    return result;
  };

  // initialize reduction with the first value
  minmax_pair<int> init = unary_op(data[0]);

  // compute minimum and maximum values
  minmax_pair<int> result = thrust::transform_reduce(data.begin(), data.end(), unary_op, init, binary_op);

  // print results
  std::cout << "[ ";
  for (auto& e : host_data)
  {
    std::cout << e << " ";
  }
  std::cout << "]" << std::endl;

  std::cout << "minimum = " << result.min_val << std::endl;
  std::cout << "maximum = " << result.max_val << std::endl;

  return 0;
}
