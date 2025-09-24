#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

#include <cuda/functional>

#include <iostream>

// this example computes the maximum absolute difference
// between the elements of two vectors

template <typename T>
struct abs_diff
{
  __host__ __device__ T operator()(const T& a, const T& b)
  {
    return fabsf(b - a);
  }
};

int main()
{
  thrust::device_vector<float> d_a = {1.0, 2.0, 3.0, 4.0};
  thrust::device_vector<float> d_b = {2.0, 4.0, 3.0, 0.0};

  // initial value of the reduction
  float init = 0;

  // binary operations
  cuda::maximum<float> binary_op1{};
  abs_diff<float> binary_op2;

  float max_abs_diff = thrust::inner_product(d_a.begin(), d_a.end(), d_b.begin(), init, binary_op1, binary_op2);

  std::cout << "maximum absolute difference: " << max_abs_diff << std::endl;
  return 0;
}
