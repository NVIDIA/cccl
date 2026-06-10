#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <unittest/unittest.h>

using namespace unittest;

template <typename Tuple>
struct TuplePlus
{
  _CCCL_HOST_DEVICE Tuple operator()(Tuple x, Tuple y) const
  {
    return cuda::std::make_tuple(
      cuda::std::get<0>(x) + cuda::std::get<0>(y), cuda::std::get<1>(x) + cuda::std::get<1>(y));
  }
}; // end SumTuple

template <typename T>
struct TestZipIteratorReduce
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data0 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data1 = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data0 = h_data0;
    thrust::device_vector<T> d_data1 = h_data1;

    using Tuple = cuda::std::tuple<T, T>;

    // run on host
    Tuple h_result = thrust::reduce(
      thrust::make_zip_iterator(h_data0.begin(), h_data1.begin()),
      thrust::make_zip_iterator(h_data0.end(), h_data1.end()),
      cuda::std::make_tuple<T, T>(0, 0),
      TuplePlus<Tuple>());

    // run on device
    Tuple d_result = thrust::reduce(
      thrust::make_zip_iterator(d_data0.begin(), d_data1.begin()),
      thrust::make_zip_iterator(d_data0.end(), d_data1.end()),
      ::cuda::std::make_tuple<T, T>(0, 0),
      TuplePlus<Tuple>());

    ASSERT_EQUAL(cuda::std::get<0>(h_result), cuda::std::get<0>(d_result));
    ASSERT_EQUAL(cuda::std::get<1>(h_result), cuda::std::get<1>(d_result));
  }
};
VariableUnitTest<TestZipIteratorReduce, IntegralTypes> TestZipIteratorReduceInstance;
