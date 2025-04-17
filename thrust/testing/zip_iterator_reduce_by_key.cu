#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

#include <unittest/unittest.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  include <unittest/cuda/testframework.h>
#endif

using namespace unittest;

template <typename Tuple>
struct TuplePlus
{
  _CCCL_HOST_DEVICE Tuple operator()(Tuple x, Tuple y) const
  {
    using namespace thrust;
    return make_tuple(get<0>(x) + get<0>(y), get<1>(x) + get<1>(y));
  }
}; // end TuplePlus

template <typename T>
struct TestZipIteratorReduceByKey
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_integers<bool>(n);
    host_vector<T> h_data1 = unittest::random_integers<T>(n);
    host_vector<T> h_data2 = unittest::random_integers<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    using Tuple = tuple<T, T>;

    // integer key, tuple value
    {
      host_vector<T> h_data3(n, 0);
      host_vector<T> h_data4(n, 0);
      host_vector<T> h_data5(n, 0);
      device_vector<T> d_data3(n, 0);
      device_vector<T> d_data4(n, 0);
      device_vector<T> d_data5(n, 0);

      // run on host
      reduce_by_key(
        h_data0.begin(),
        h_data0.end(),
        make_zip_iterator(h_data1.begin(), h_data2.begin()),
        h_data3.begin(),
        make_zip_iterator(h_data4.begin(), h_data5.begin()),
        ::cuda::std::equal_to<T>(),
        TuplePlus<Tuple>());

      // run on device
      reduce_by_key(
        d_data0.begin(),
        d_data0.end(),
        make_zip_iterator(d_data1.begin(), d_data2.begin()),
        d_data3.begin(),
        make_zip_iterator(d_data4.begin(), d_data5.begin()),
        ::cuda::std::equal_to<T>(),
        TuplePlus<Tuple>());

      ASSERT_EQUAL(h_data3, d_data3);
      ASSERT_EQUAL(h_data4, d_data4);
      ASSERT_EQUAL(h_data5, d_data5);
    }

    // The tests below get miscompiled on Tesla hw for 8b types

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    if (const CUDATestDriver* driver = dynamic_cast<const CUDATestDriver*>(&UnitTestDriver::s_driver()))
    {
      if (typeid(T) == typeid(unittest::uint8_t) && driver->current_device_architecture() < 200)
      {
        KNOWN_FAILURE;
      } // end if
    } // end if
#endif

    // tuple key, tuple value
    {
      host_vector<T> h_data3(n, 0);
      host_vector<T> h_data4(n, 0);
      host_vector<T> h_data5(n, 0);
      host_vector<T> h_data6(n, 0);
      device_vector<T> d_data3(n, 0);
      device_vector<T> d_data4(n, 0);
      device_vector<T> d_data5(n, 0);
      device_vector<T> d_data6(n, 0);

      // run on host
      reduce_by_key(
        make_zip_iterator(h_data0.begin(), h_data0.begin()),
        make_zip_iterator(h_data0.end(), h_data0.end()),
        make_zip_iterator(h_data1.begin(), h_data2.begin()),
        make_zip_iterator(h_data3.begin(), h_data4.begin()),
        make_zip_iterator(h_data5.begin(), h_data6.begin()),
        ::cuda::std::equal_to<Tuple>(),
        TuplePlus<Tuple>());

      // run on device
      reduce_by_key(
        make_zip_iterator(d_data0.begin(), d_data0.begin()),
        make_zip_iterator(d_data0.end(), d_data0.end()),
        make_zip_iterator(d_data1.begin(), d_data2.begin()),
        make_zip_iterator(d_data3.begin(), d_data4.begin()),
        make_zip_iterator(d_data5.begin(), d_data6.begin()),
        ::cuda::std::equal_to<Tuple>(),
        TuplePlus<Tuple>());

      ASSERT_EQUAL(h_data3, d_data3);
      ASSERT_EQUAL(h_data4, d_data4);
      ASSERT_EQUAL(h_data5, d_data5);
      ASSERT_EQUAL(h_data6, d_data6);
    }

    // const inputs, see #1527
    {
      host_vector<float> h_data3(n, 0.0f);
      host_vector<T> h_data4(n, 0);
      host_vector<T> h_data5(n, 0);
      host_vector<float> h_data6(n, 0.0f);
      device_vector<float> d_data3(n, 0.0f);
      device_vector<T> d_data4(n, 0);
      device_vector<T> d_data5(n, 0);
      device_vector<float> d_data6(n, 0.0f);

      // run on host
      const T* h_begin1     = thrust::raw_pointer_cast(h_data1.data());
      const T* h_begin2     = thrust::raw_pointer_cast(h_data2.data());
      const float* h_begin3 = thrust::raw_pointer_cast(h_data3.data());
      T* h_begin4           = thrust::raw_pointer_cast(h_data4.data());
      T* h_begin5           = thrust::raw_pointer_cast(h_data5.data());
      float* h_begin6       = thrust::raw_pointer_cast(h_data6.data());
      thrust::reduce_by_key(
        thrust::host,
        thrust::make_zip_iterator(h_begin1, h_begin2),
        thrust::make_zip_iterator(h_begin1, h_begin2) + n,
        h_begin3,
        thrust::make_zip_iterator(h_begin4, h_begin5),
        h_begin6);

      // run on device
      const T* d_begin1     = thrust::raw_pointer_cast(d_data1.data());
      const T* d_begin2     = thrust::raw_pointer_cast(d_data2.data());
      const float* d_begin3 = thrust::raw_pointer_cast(d_data3.data());
      T* d_begin4           = thrust::raw_pointer_cast(d_data4.data());
      T* d_begin5           = thrust::raw_pointer_cast(d_data5.data());
      float* d_begin6       = thrust::raw_pointer_cast(d_data6.data());
      thrust::reduce_by_key(
        thrust::device,
        thrust::make_zip_iterator(d_begin1, d_begin2),
        thrust::make_zip_iterator(d_begin1, d_begin2) + n,
        d_begin3,
        thrust::make_zip_iterator(d_begin4, d_begin5),
        d_begin6);

      ASSERT_EQUAL(h_data3, d_data3);
      ASSERT_EQUAL(h_data4, d_data4);
      ASSERT_EQUAL(h_data5, d_data5);
      ASSERT_EQUAL(h_data6, d_data6);
    }
  }
};
VariableUnitTest<TestZipIteratorReduceByKey, UnsignedIntegralTypes> TestZipIteratorReduceByKeyInstance;
