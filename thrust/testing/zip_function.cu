#include <thrust/detail/config.h>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

#include <iostream>

#include <unittest/unittest.h>

using namespace unittest;

struct SumThree
{
  template <typename T1, typename T2, typename T3>
  _CCCL_HOST_DEVICE auto operator()(T1 x, T2 y, T3 z) const THRUST_DECLTYPE_RETURNS(x + y + z)
}; // end SumThree

struct SumThreeTuple
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE auto operator()(Tuple x) const
    THRUST_DECLTYPE_RETURNS(cuda::std::get<0>(x) + cuda::std::get<1>(x) + cuda::std::get<2>(x))
}; // end SumThreeTuple

template <typename T>
struct TestZipFunctionCtor
{
  void operator()()
  {
    ASSERT_EQUAL(thrust::zip_function<SumThree>()(cuda::std::tuple(1, 2, 3)), SumThree{}(1, 2, 3));
    ASSERT_EQUAL(thrust::zip_function<SumThree>(SumThree{})(cuda::std::tuple(1, 2, 3)), SumThree{}(1, 2, 3));
#ifdef __cpp_deduction_guides
    ASSERT_EQUAL(thrust::zip_function(SumThree{})(cuda::std::tuple(1, 2, 3)), SumThree{}(1, 2, 3));
#endif // __cpp_deduction_guides
  }
};
SimpleUnitTest<TestZipFunctionCtor, type_list<int>> TestZipFunctionCtorInstance;

template <typename T>
struct TestZipFunctionTransform
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data0 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data1 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data2 = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data0 = h_data0;
    thrust::device_vector<T> d_data1 = h_data1;
    thrust::device_vector<T> d_data2 = h_data2;

    thrust::host_vector<T> h_result_tuple(n);
    thrust::host_vector<T> h_result_zip(n);
    thrust::device_vector<T> d_result_zip(n);

    // Tuple base case

    thrust::transform(thrust::make_zip_iterator(h_data0.begin(), h_data1.begin(), h_data2.begin()),
                      thrust::make_zip_iterator(h_data0.end(), h_data1.end(), h_data2.end()),
                      h_result_tuple.begin(),
                      SumThreeTuple{});
    // Zip Function
    thrust::transform(thrust::make_zip_iterator(h_data0.begin(), h_data1.begin(), h_data2.begin()),
                      thrust::make_zip_iterator(h_data0.end(), h_data1.end(), h_data2.end()),
                      h_result_zip.begin(),
                      thrust::make_zip_function(SumThree{}));
    thrust::transform(thrust::make_zip_iterator(d_data0.begin(), d_data1.begin(), d_data2.begin()),
                      thrust::make_zip_iterator(d_data0.end(), d_data1.end(), d_data2.end()),
                      d_result_zip.begin(),
                      thrust::make_zip_function(SumThree{}));

    ASSERT_EQUAL(h_result_tuple, h_result_zip);
    ASSERT_EQUAL(h_result_tuple, d_result_zip);
  }
};
VariableUnitTest<TestZipFunctionTransform, ThirtyTwoBitTypes> TestZipFunctionTransformInstance;

struct RemovePred
{
  _CCCL_HOST_DEVICE bool operator()(const cuda::std::tuple<uint32_t, uint32_t>& ele1, const float&)
  {
    return cuda::std::get<0>(ele1) == cuda::std::get<1>(ele1);
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
  _CCCL_HOST_DEVICE bool operator()(
    const cuda::std::tuple<uint32_t, cuda::std::tuple<cuda::std::tuple<int, int>, cuda::std::tuple<int, int>>>& idAndPt)
  {
    cuda::std::tuple<cuda::std::tuple<int, int>, cuda::std::tuple<int, int>> ele1 = cuda::std::get<1>(idAndPt);
    cuda::std::tuple<int, int> p1                                                 = cuda::std::get<0>(ele1);
    cuda::std::tuple<int, int> p2                                                 = cuda::std::get<1>(ele1);
    return cuda::std::get<0>(p1) == cuda::std::get<0>(p2) || cuda::std::get<1>(p1) == cuda::std::get<1>(p2);
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
  _CCCL_DEVICE _CCCL_FORCEINLINE bool operator()(const cuda::std::tuple<cuda::std::tuple<int, int>, int>& a,
                                                 const cuda::std::tuple<cuda::std::tuple<int, int>, int>& b)
  {
    return cuda::std::get<1>(a) < cuda::std::get<1>(b);
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
