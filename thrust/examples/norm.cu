#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <iostream>

//   This example computes the norm [1] of a vector.  The norm is
// computed by squaring all numbers in the vector, summing the
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the
// transform_reduce() algorithm.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm

int main()
{
  // initialize device vector directly
  thrust::device_vector<float> d_x = {1.0, 2.0, 3.0, 4.0};

  // lambda that computes the square of a number f(x) -> x*x
  auto square = [] __device__(const float& x) {
    return x * x;
  };

  // setup arguments
  cuda::std::plus<float> binary_op;
  float init = 0;

  // compute norm
  float norm = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), square, init, binary_op));

  std::cout << "norm is " << norm << std::endl;

  return 0;
}
