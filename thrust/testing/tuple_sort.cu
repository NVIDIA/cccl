#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <unittest/unittest.h>

using namespace unittest;

struct MakeTupleFunctor
{
  template <typename T1, typename T2>
  _CCCL_HOST_DEVICE thrust::tuple<T1, T2> operator()(T1& lhs, T2& rhs)
  {
    return thrust::make_tuple(lhs, rhs);
  }
};

template <int N>
struct GetFunctor
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE typename thrust::tuple_element<N, Tuple>::type operator()(const Tuple& t)
  {
    return thrust::get<N>(t);
  }
};

template <typename T>
struct TestTupleStableSort
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_keys   = random_integers<T>(n);
    host_vector<T> h_values = random_integers<T>(n);

    // zip up the data
    host_vector<tuple<T, T>> h_tuples(n);
    thrust::transform(h_keys.begin(), h_keys.end(), h_values.begin(), h_tuples.begin(), MakeTupleFunctor());

    // copy to device
    device_vector<tuple<T, T>> d_tuples = h_tuples;

    // sort on host
    thrust::stable_sort(h_tuples.begin(), h_tuples.end());

    // sort on device
    thrust::stable_sort(d_tuples.begin(), d_tuples.end());

    ASSERT_EQUAL(true, thrust::is_sorted(d_tuples.begin(), d_tuples.end()));

    // select keys
    thrust::transform(h_tuples.begin(), h_tuples.end(), h_keys.begin(), GetFunctor<0>());

    device_vector<T> d_keys(h_keys.size());
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_keys.begin(), GetFunctor<0>());

    // select values
    thrust::transform(h_tuples.begin(), h_tuples.end(), h_values.begin(), GetFunctor<1>());

    device_vector<T> d_values(h_values.size());
    thrust::transform(d_tuples.begin(), d_tuples.end(), d_values.begin(), GetFunctor<1>());

    ASSERT_ALMOST_EQUAL(h_keys, d_keys);
    ASSERT_ALMOST_EQUAL(h_values, d_values);
  }
};
DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TestTupleStableSort,
                                          unittest::type_list<unittest::int8_t, unittest::int16_t, unittest::int32_t>);
