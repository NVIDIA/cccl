#include <thrust/detail/config.h>

#if !defined(THRUST_LEGACY_GCC)

#  include <thrust/device_vector.h>
#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/remove.h>
#  include <thrust/sort.h>
#  include <thrust/transform.h>
#  include <thrust/zip_function.h>

#  include <iostream>

#  include <unittest/unittest.h>

using namespace unittest;

struct SumThree
{
  template <typename T1, typename T2, typename T3>
  __host__ __device__ auto operator()(T1 x, T2 y, T3 z) const THRUST_DECLTYPE_RETURNS(x + y + z)
}; // end SumThree

struct SumThreeTuple
{
  template <typename Tuple>
  __host__ __device__ auto operator()(Tuple x) const
    THRUST_DECLTYPE_RETURNS(thrust::get<0>(x) + thrust::get<1>(x) + thrust::get<2>(x))
}; // end SumThreeTuple

template <typename T>
struct TestZipFunctionTransform
{
  void operator()(const size_t n)
  {
    using namespace thrust;

    host_vector<T> h_data0 = unittest::random_samples<T>(n);
    host_vector<T> h_data1 = unittest::random_samples<T>(n);
    host_vector<T> h_data2 = unittest::random_samples<T>(n);

    device_vector<T> d_data0 = h_data0;
    device_vector<T> d_data1 = h_data1;
    device_vector<T> d_data2 = h_data2;

    host_vector<T> h_result_tuple(n);
    host_vector<T> h_result_zip(n);
    device_vector<T> d_result_zip(n);

    // Tuple base case
    transform(make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin(), h_data2.begin())),
              make_zip_iterator(make_tuple(h_data0.end(), h_data1.end(), h_data2.end())),
              h_result_tuple.begin(),
              SumThreeTuple{});
    // Zip Function
    transform(make_zip_iterator(make_tuple(h_data0.begin(), h_data1.begin(), h_data2.begin())),
              make_zip_iterator(make_tuple(h_data0.end(), h_data1.end(), h_data2.end())),
              h_result_zip.begin(),
              make_zip_function(SumThree{}));
    transform(make_zip_iterator(make_tuple(d_data0.begin(), d_data1.begin(), d_data2.begin())),
              make_zip_iterator(make_tuple(d_data0.end(), d_data1.end(), d_data2.end())),
              d_result_zip.begin(),
              make_zip_function(SumThree{}));

    ASSERT_EQUAL(h_result_tuple, h_result_zip);
    ASSERT_EQUAL(h_result_tuple, d_result_zip);
  }
};
VariableUnitTest<TestZipFunctionTransform, ThirtyTwoBitTypes> TestZipFunctionTransformInstance;

struct RemovePred
{
  __host__ __device__ bool operator()(const thrust::tuple<uint32_t, uint32_t>& ele1, const float&)
  {
    return thrust::get<0>(ele1) == thrust::get<1>(ele1);
  }
};
template <typename T>
struct TestZipFunctionMixed
{
  void operator()()
  {
    thrust::device_vector<uint32_t> vecA{0, 0, 2, 0};
    thrust::device_vector<uint32_t> vecB{0, 2, 2, 2};
    thrust::device_vector<float> vecC{88.0f, 88.0f, 89.0f, 89.0f};
    thrust::device_vector<float> expected{88.0f, 89.0f};

    auto inputKeyItBegin =
      thrust::make_zip_iterator(thrust::make_zip_iterator(vecA.begin(), vecB.begin()), vecC.begin());
    auto endIt =
      thrust::remove_if(inputKeyItBegin, inputKeyItBegin + vecA.size(), thrust::make_zip_function(RemovePred{}));
    auto numEle = endIt - inputKeyItBegin;
    vecA.resize(numEle);
    vecB.resize(numEle);
    vecC.resize(numEle);

    ASSERT_EQUAL(numEle, 2);
    ASSERT_EQUAL(vecC, expected);
  }
};
SimpleUnitTest<TestZipFunctionMixed, type_list<int, float>> TestZipFunctionMixedInstance;

struct NestedFunctionCall
{
  __host__ __device__ bool
  operator()(const thrust::tuple<uint32_t, thrust::tuple<thrust::tuple<int, int>, thrust::tuple<int, int>>>& idAndPt)
  {
    thrust::tuple<thrust::tuple<int, int>, thrust::tuple<int, int>> ele1 = thrust::get<1>(idAndPt);
    thrust::tuple<int, int> p1                                           = thrust::get<0>(ele1);
    thrust::tuple<int, int> p2                                           = thrust::get<1>(ele1);
    return thrust::get<0>(p1) == thrust::get<0>(p2) || thrust::get<1>(p1) == thrust::get<1>(p2);
  }
};

template <typename T>
struct TestNestedZipFunction
{
  void operator()()
  {
    thrust::device_vector<int> PX{0, 1, 2, 3};
    thrust::device_vector<int> PY{0, 1, 2, 2};
    thrust::device_vector<uint32_t> SS{0, 1, 2};
    thrust::device_vector<uint32_t> ST{1, 2, 3};
    thrust::device_vector<float> vecC{88.0f, 88.0f, 89.0f, 89.0f};

    auto segIt = thrust::make_zip_iterator(
      thrust::make_zip_iterator(thrust::make_permutation_iterator(PX.begin(), SS.begin()),
                                thrust::make_permutation_iterator(PY.begin(), SS.begin())),
      thrust::make_zip_iterator(thrust::make_permutation_iterator(PX.begin(), ST.begin()),
                                thrust::make_permutation_iterator(PY.begin(), ST.begin())));
    auto idAndSegIt = thrust::make_zip_iterator(thrust::make_counting_iterator(0u), segIt);

    thrust::device_vector<bool> isMH{false, false, false};
    thrust::device_vector<bool> expected{false, false, true};
    thrust::transform(idAndSegIt, idAndSegIt + SS.size(), isMH.begin(), NestedFunctionCall{});
    ASSERT_EQUAL(isMH, expected);
  }
};
SimpleUnitTest<TestNestedZipFunction, type_list<int, float>> TestNestedZipFunctionInstance;

struct SortPred
{
  __device__ __forceinline__ bool
  operator()(const thrust::tuple<thrust::tuple<int, int>, int>& a, const thrust::tuple<thrust::tuple<int, int>, int>& b)
  {
    return thrust::get<1>(a) < thrust::get<1>(b);
  }
};
template <typename T>
struct TestNestedZipFunction2
{
  void operator()()
  {
    thrust::device_vector<int> A(5);
    thrust::device_vector<int> B(5);
    thrust::device_vector<int> C(5);
    auto n = A.size();

    auto tupleIt       = thrust::make_zip_iterator(cuda::std::begin(A), cuda::std::begin(B));
    auto nestedTupleIt = thrust::make_zip_iterator(tupleIt, cuda::std::begin(C));
    thrust::sort(nestedTupleIt, nestedTupleIt + n, SortPred{});
  }
};
SimpleUnitTest<TestNestedZipFunction2, type_list<int, float>> TestNestedZipFunctionInstance2;
#endif // _CCCL_STD_VER
