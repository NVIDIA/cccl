#include <thrust/transform.h>

#include <cuda/std/tuple>

#include <unittest/unittest.h>

using namespace unittest;

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE cuda::std::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return cuda::std::tuple(lhs, rhs);
  }
};

template <int N>
struct GetFunctor
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE typename cuda::std::tuple_element<N, Tuple>::type operator()(const Tuple& t)
  {
    return cuda::std::get<N>(t);
  }
};

template <typename T>
struct TestTupleTransform
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_t1 = random_integers<T>(n);
    thrust::host_vector<T> h_t2 = random_integers<T>(n);

    // zip up the data
    thrust::host_vector<cuda::std::tuple<T, T>> h_tuples(n);
    thrust::transform(h_t1.begin(), h_t1.end(), h_t2.begin(), h_tuples.begin(), MakeTupleFunctor());

    // copy to device
    thrust::device_vector<cuda::std::tuple<T, T>> d_tuples = h_tuples;

    thrust::device_vector<T> d_t1(n), d_t2(n);

    // select 0th
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t1.begin(), GetFunctor<0>());

    // select 1st
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_t2.begin(), GetFunctor<1>());

    ASSERT_ALMOST_EQUAL(h_t1, d_t1);
    ASSERT_ALMOST_EQUAL(h_t2, d_t2);

    ASSERT_EQUAL_QUIET(h_tuples, d_tuples);
  }
};
VariableUnitTest<TestTupleTransform, SignedIntegralTypes> TestTupleTransformInstance;
