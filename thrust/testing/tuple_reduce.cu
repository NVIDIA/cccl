#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cuda/std/tuple>

#include <unittest/unittest.h>

using namespace unittest;

struct SumTupleFunctor
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE Tuple operator()(const Tuple& lhs, const Tuple& rhs)
  {
    using cuda::std::get;

    return cuda::std::tuple(get<0>(lhs) + get<0>(rhs), get<1>(lhs) + get<1>(rhs));
  }
};

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE cuda::std::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return cuda::std::tuple(lhs, rhs);
  }
};

template <typename T>
struct TestTupleReduce
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

    cuda::std::tuple<T, T> zero(0, 0);

    // sum on host
    cuda::std::tuple<T, T> h_result = thrust::reduce(h_tuples.begin(), h_tuples.end(), zero, SumTupleFunctor());

    // sum on device
    cuda::std::tuple<T, T> d_result = thrust::reduce(d_tuples.begin(), d_tuples.end(), zero, SumTupleFunctor());

    ASSERT_EQUAL_QUIET(h_result, d_result);
  }
};
VariableUnitTest<TestTupleReduce, IntegralTypes> TestTupleReduceInstance;
