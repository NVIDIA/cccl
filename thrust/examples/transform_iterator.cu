#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <iostream>
#include <iterator>
#include <string>

// helper lambda to clamp a value to the range [lo, hi]
template <typename T>
auto make_clamp_lambda(T lo, T hi)
{
  return [=] __device__(T x) {
    if (x < lo)
    {
      return lo;
    }
    else if (x < hi)
    {
      return x;
    }
    else
    {
      return hi;
    }
  };
}

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
  using T = typename std::iterator_traits<Iterator>::value_type;

  std::cout << name << ": ";
  thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));
  std::cout << "\n";
}

int main()
{
  // clamp values to the range [1, 5]
  int lo = 1;
  int hi = 5;

  // define some types
  using Vector         = thrust::device_vector<int>;
  using VectorIterator = Vector::iterator;

  // initialize values
  Vector values(8);

  values[0] = 2;
  values[1] = 5;
  values[2] = 7;
  values[3] = 1;
  values[4] = 6;
  values[5] = 0;
  values[6] = 3;
  values[7] = 8;

  print_range("values         ", values.begin(), values.end());

  // create a lambda that clamps values to [lo, hi]
  auto clamp_fn = make_clamp_lambda(lo, hi);

  // create a transform_iterator that applies the clamp lambda to the values array
  auto cv_begin = thrust::make_transform_iterator(values.begin(), clamp_fn);
  auto cv_end   = cv_begin + values.size();

  // now [clamped_begin, clamped_end) defines a sequence of clamped values
  print_range("clamped values ", cv_begin, cv_end);

  ////
  // compute the sum of the clamped sequence with reduce()
  std::cout << "sum of clamped values : " << thrust::reduce(cv_begin, cv_end) << "\n";

  ////
  // combine transform_iterator with other fancy iterators like counting_iterator
  using CountingIterator = thrust::counting_iterator<int>;

  CountingIterator count_begin(0);
  CountingIterator count_end(10);

  print_range("sequence         ", count_begin, count_end);

  auto cs_begin = thrust::make_transform_iterator(count_begin, clamp_fn);
  auto cs_end   = thrust::make_transform_iterator(count_end, clamp_fn);

  print_range("clamped sequence ", cs_begin, cs_end);

  ////
  // combine transform_iterator with another transform_iterator
  auto ncs_begin = thrust::make_transform_iterator(cs_begin, cuda::std::negate<int>());
  auto ncs_end   = thrust::make_transform_iterator(cs_end, cuda::std::negate<int>());

  print_range("negated sequence ", ncs_begin, ncs_end);

  ////
  // using a simple negate lambda
  auto simple_negate = [] __device__(int x) {
    return -x;
  };

  auto nv_begin = thrust::make_transform_iterator(values.begin(), simple_negate);
  auto nv_end   = thrust::make_transform_iterator(values.end(), simple_negate);

  print_range("negated values ", nv_begin, nv_end);

  return 0;
}
