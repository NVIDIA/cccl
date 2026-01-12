#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda/std/type_traits>

#include <unittest/unittest.h>

using namespace unittest;

// ensure that we properly support thrust::zip_iterator from cuda::std
void TestZipIteratorTraits()
{
  using base_it = thrust::host_vector<int>::iterator;

  using it        = thrust::zip_iterator<cuda::std::tuple<base_it, base_it>>;
  using traits    = cuda::std::iterator_traits<it>;
  using reference = thrust::detail::tuple_of_iterator_references<int&, int&>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, cuda::std::tuple<int, int>>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);

  static_assert(cuda::std::is_same_v<traits::reference, reference>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, ::cuda::std::random_access_iterator_tag>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__has_random_access_traversal<it>);

  static_assert(!cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}
DECLARE_UNITTEST(TestZipIteratorTraits);

template <typename T>
struct TestZipIteratorConstructionFromIterators
{
  template <typename Vector>
  void test()
  {
    Vector v0(4);
    Vector v1(4);
    Vector v2(4);

    // initialize input
    thrust::sequence(v0.begin(), v0.end());
    thrust::sequence(v1.begin(), v1.end());
    thrust::sequence(v2.begin(), v2.end());

    using IteratorTuple = cuda::std::tuple<typename Vector::iterator, typename Vector::iterator>;
    using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

    // test construction
    thrust::zip_iterator iter0(v0.begin(), v1.begin());
    ASSERT_EQUAL(true, iter0 == ZipIterator{cuda::std::make_tuple(v0.begin(), v1.begin())});
  }

  void operator()(void)
  {
    test<thrust::host_vector<T>>();
    test<thrust::device_vector<T>>();
  }
};
SimpleUnitTest<TestZipIteratorConstructionFromIterators, type_list<int>>
  TestZipIteratorConstructionFromIteratorsInstance;

template <typename T>
struct TestZipIteratorManipulation
{
  template <typename Vector>
  void test()
  {
    Vector v0(4);
    Vector v1(4);
    Vector v2(4);

    // initialize input
    thrust::sequence(v0.begin(), v0.end());
    thrust::sequence(v1.begin(), v1.end());
    thrust::sequence(v2.begin(), v2.end());

    using IteratorTuple = cuda::std::tuple<typename Vector::iterator, typename Vector::iterator>;
    IteratorTuple t     = cuda::std::make_tuple(v0.begin(), v1.begin());
    using ZipIterator   = thrust::zip_iterator<IteratorTuple>;

    // test construction from tuple
    ZipIterator iter0 = thrust::make_zip_iterator(t);
    ASSERT_EQUAL(true, iter0 == ZipIterator{t});
    ASSERT_EQUAL_QUIET(v0.begin(), cuda::std::get<0>(iter0.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin(), cuda::std::get<1>(iter0.get_iterator_tuple()));
    static_assert(cuda::std::is_same_v<decltype(thrust::zip_iterator{t}), ZipIterator>); // CTAD

    // test construction from pack
    ZipIterator iter0_pack = thrust::make_zip_iterator(v0.begin(), v1.begin());
    ASSERT_EQUAL(true, (iter0_pack == ZipIterator{v0.begin(), v1.begin()}));
    ASSERT_EQUAL_QUIET(v0.begin(), cuda::std::get<0>(iter0_pack.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin(), cuda::std::get<1>(iter0_pack.get_iterator_tuple()));
    static_assert(cuda::std::is_same_v<decltype(thrust::zip_iterator{v0.begin(), v1.begin()}), ZipIterator>); // CTAD

    // test dereference
    ASSERT_EQUAL(*v0.begin(), cuda::std::get<0>(*iter0));
    ASSERT_EQUAL(*v1.begin(), cuda::std::get<1>(*iter0));

    // test equality
    ZipIterator iter1 = iter0;
    ZipIterator iter2 = thrust::make_zip_iterator(v0.begin(), v2.begin());
    ZipIterator iter3 = thrust::make_zip_iterator(v1.begin(), v2.begin());
    ASSERT_EQUAL(true, iter0 == iter1);
    ASSERT_EQUAL(true, iter0 == iter2);
    ASSERT_EQUAL(false, iter0 == iter3);

    // test inequality
    ASSERT_EQUAL(false, iter0 != iter1);
    ASSERT_EQUAL(false, iter0 != iter2);
    ASSERT_EQUAL(true, iter0 != iter3);

    // test advance
    ZipIterator iter4 = iter0 + 1;
    ASSERT_EQUAL_QUIET(v0.begin() + 1, cuda::std::get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 1, cuda::std::get<1>(iter4.get_iterator_tuple()));

    // test pre-increment
    ++iter4;
    ASSERT_EQUAL_QUIET(v0.begin() + 2, cuda::std::get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 2, cuda::std::get<1>(iter4.get_iterator_tuple()));

    // test post-increment
    iter4++;
    ASSERT_EQUAL_QUIET(v0.begin() + 3, cuda::std::get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 3, cuda::std::get<1>(iter4.get_iterator_tuple()));

    // test pre-decrement
    --iter4;
    ASSERT_EQUAL_QUIET(v0.begin() + 2, cuda::std::get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 2, cuda::std::get<1>(iter4.get_iterator_tuple()));

    // test post-decrement
    iter4--;
    ASSERT_EQUAL_QUIET(v0.begin() + 1, cuda::std::get<0>(iter4.get_iterator_tuple()));
    ASSERT_EQUAL_QUIET(v1.begin() + 1, cuda::std::get<1>(iter4.get_iterator_tuple()));

    // test difference
    ASSERT_EQUAL(1, iter4 - iter0);
    ASSERT_EQUAL(-1, iter0 - iter4);
  }

  void operator()(void)
  {
    test<thrust::host_vector<T>>();
    test<thrust::device_vector<T>>();
  }
};
SimpleUnitTest<TestZipIteratorManipulation, type_list<int>> TestZipIteratorManipulationInstance;
static_assert(cuda::std::is_trivially_copy_constructible<thrust::zip_iterator<cuda::std::tuple<int*, int*>>>::value,
              "");

template <typename T>
struct TestZipIteratorReference
{
  void operator()(void)
  {
    // test host types
    using Iterator1      = typename thrust::host_vector<T>::iterator;
    using Iterator2      = typename thrust::host_vector<T>::const_iterator;
    using IteratorTuple1 = cuda::std::tuple<Iterator1, Iterator2>;
    using ZipIterator1   = thrust::zip_iterator<IteratorTuple1>;

    using zip_iterator_reference_type1 = thrust::detail::it_reference_t<ZipIterator1>;

    thrust::host_vector<T> h_variable(1);

    using reference_type1 = cuda::std::tuple<T&, const T&>;

    reference_type1 ref1(*h_variable.begin(), *h_variable.cbegin());
    zip_iterator_reference_type1 test1(*h_variable.begin(), *h_variable.cbegin());

    ASSERT_EQUAL_QUIET(ref1, test1);
    ASSERT_EQUAL(cuda::std::get<0>(ref1), cuda::std::get<0>(test1));
    ASSERT_EQUAL(cuda::std::get<1>(ref1), cuda::std::get<1>(test1));

    // test device types
    using Iterator3      = typename thrust::device_vector<T>::iterator;
    using Iterator4      = typename thrust::device_vector<T>::const_iterator;
    using IteratorTuple2 = cuda::std::tuple<Iterator3, Iterator4>;
    using ZipIterator2   = thrust::zip_iterator<IteratorTuple2>;

    using zip_iterator_reference_type2 = thrust::detail::it_reference_t<ZipIterator2>;

    thrust::device_vector<T> d_variable(1);

    using reference_type2 = cuda::std::tuple<thrust::device_reference<T>, thrust::device_reference<const T>>;

    reference_type2 ref2(*d_variable.begin(), *d_variable.cbegin());
    zip_iterator_reference_type2 test2(*d_variable.begin(), *d_variable.cbegin());

    ASSERT_EQUAL_QUIET(ref2, test2);
    ASSERT_EQUAL(cuda::std::get<0>(ref2), cuda::std::get<0>(test2));
    ASSERT_EQUAL(cuda::std::get<1>(ref2), cuda::std::get<1>(test2));
  } // end operator()()
};
SimpleUnitTest<TestZipIteratorReference, NumericTypes> TestZipIteratorReferenceInstance;

template <typename Vector>
void TestZipIteratorCopy()
{
  using T = typename Vector::value_type;

  Vector input0(4), input1(4);
  Vector output0(4), output1(4);

  // initialize input
  thrust::sequence(input0.begin(), input0.end(), T{0});
  thrust::sequence(input1.begin(), input1.end(), T{13});

  thrust::copy(thrust::make_zip_iterator(input0.begin(), input1.begin()),
               thrust::make_zip_iterator(input0.end(), input1.end()),
               thrust::make_zip_iterator(output0.begin(), output1.begin()));

  ASSERT_EQUAL(input0, output0);
  ASSERT_EQUAL(input1, output1);
}
DECLARE_VECTOR_UNITTEST(TestZipIteratorCopy);

struct SumTwoTuple
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE cuda::std::remove_reference_t<cuda::std::tuple_element_t<0, Tuple>> operator()(Tuple x) const
  {
    return cuda::std::get<0>(x) + cuda::std::get<1>(x);
  }
}; // end SumTwoTuple

struct SumThreeTuple
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE cuda::std::remove_reference_t<cuda::std::tuple_element_t<0, Tuple>> operator()(Tuple x) const
  {
    return cuda::std::get<0>(x) + cuda::std::get<1>(x) + cuda::std::get<2>(x);
  }
}; // end SumThreeTuple

template <typename T>
struct TestZipIteratorTransform
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data0 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data1 = unittest::random_samples<T>(n);
    thrust::host_vector<T> h_data2 = unittest::random_samples<T>(n);

    thrust::device_vector<T> d_data0 = h_data0;
    thrust::device_vector<T> d_data1 = h_data1;
    thrust::device_vector<T> d_data2 = h_data2;

    thrust::host_vector<T> h_result(n);
    thrust::device_vector<T> d_result(n);

    // Tuples with 2 elements
    thrust::transform(thrust::make_zip_iterator(h_data0.begin(), h_data1.begin()),
                      thrust::make_zip_iterator(h_data0.end(), h_data1.end()),
                      h_result.begin(),
                      SumTwoTuple());
    thrust::transform(thrust::make_zip_iterator(d_data0.begin(), d_data1.begin()),
                      thrust::make_zip_iterator(d_data0.end(), d_data1.end()),
                      d_result.begin(),
                      SumTwoTuple());
    ASSERT_EQUAL(h_result, d_result);

    // Tuples with 3 elements
    thrust::transform(thrust::make_zip_iterator(h_data0.begin(), h_data1.begin(), h_data2.begin()),
                      thrust::make_zip_iterator(h_data0.end(), h_data1.end(), h_data2.end()),
                      h_result.begin(),
                      SumThreeTuple());
    thrust::transform(thrust::make_zip_iterator(d_data0.begin(), d_data1.begin(), d_data2.begin()),
                      thrust::make_zip_iterator(d_data0.end(), d_data1.end(), d_data2.end()),
                      d_result.begin(),
                      SumThreeTuple());
    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestZipIteratorTransform, ThirtyTwoBitTypes> TestZipIteratorTransformInstance;

void TestZipIteratorCopyAoSToSoA()
{
  const size_t n = 1;

  using structure                  = cuda::std::tuple<int, int>;
  using host_array_of_structures   = thrust::host_vector<structure>;
  using device_array_of_structures = thrust::device_vector<structure>;

  using host_structure_of_arrays =
    thrust::zip_iterator<cuda::std::tuple<thrust::host_vector<int>::iterator, thrust::host_vector<int>::iterator>>;

  using device_structure_of_arrays =
    thrust::zip_iterator<cuda::std::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator>>;

  host_array_of_structures h_aos(n, cuda::std::make_tuple(7, 13));
  device_array_of_structures d_aos(n, cuda::std::make_tuple(7, 13));

  // host to host
  thrust::host_vector<int> h_field0(n), h_field1(n);
  host_structure_of_arrays h_soa = thrust::make_zip_iterator(h_field0.begin(), h_field1.begin());

  thrust::copy(h_aos.begin(), h_aos.end(), h_soa);
  ASSERT_EQUAL_QUIET(cuda::std::make_tuple(7, 13), h_soa[0]);

  // host to device
  thrust::device_vector<int> d_field0(n), d_field1(n);
  device_structure_of_arrays d_soa = thrust::make_zip_iterator(d_field0.begin(), d_field1.begin());

  thrust::copy(h_aos.begin(), h_aos.end(), d_soa);
  ASSERT_EQUAL_QUIET(cuda::std::make_tuple(7, 13), d_soa[0]);

  // device to device
  thrust::fill(d_field0.begin(), d_field0.end(), 0);
  thrust::fill(d_field1.begin(), d_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), d_soa);
  ASSERT_EQUAL_QUIET(cuda::std::make_tuple(7, 13), d_soa[0]);

  // device to host
  thrust::fill(h_field0.begin(), h_field0.end(), 0);
  thrust::fill(h_field1.begin(), h_field1.end(), 0);

  thrust::copy(d_aos.begin(), d_aos.end(), h_soa);
  ASSERT_EQUAL_QUIET(cuda::std::make_tuple(7, 13), h_soa[0]);
}
DECLARE_UNITTEST(TestZipIteratorCopyAoSToSoA);

void TestZipIteratorCopySoAToAoS()
{
  const size_t n = 1;

  using structure                  = cuda::std::tuple<int, int>;
  using host_array_of_structures   = thrust::host_vector<structure>;
  using device_array_of_structures = thrust::device_vector<structure>;

  using host_structure_of_arrays =
    thrust::zip_iterator<cuda::std::tuple<thrust::host_vector<int>::iterator, thrust::host_vector<int>::iterator>>;

  using device_structure_of_arrays =
    thrust::zip_iterator<cuda::std::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator>>;

  thrust::host_vector<int> h_field0(n, 7), h_field1(n, 13);
  thrust::device_vector<int> d_field0(n, 7), d_field1(n, 13);

  host_structure_of_arrays h_soa   = thrust::make_zip_iterator(h_field0.begin(), h_field1.begin());
  device_structure_of_arrays d_soa = thrust::make_zip_iterator(d_field0.begin(), d_field1.begin());

  host_array_of_structures h_aos(n);
  device_array_of_structures d_aos(n);

  // host to host
  thrust::fill(h_aos.begin(), h_aos.end(), cuda::std::make_tuple(0, 0));

  thrust::copy(h_soa, h_soa + n, h_aos.begin());
  ASSERT_EQUAL_QUIET(7, cuda::std::get<0>(h_soa[0]));
  ASSERT_EQUAL_QUIET(13, cuda::std::get<1>(h_soa[0]));

  // host to device
  thrust::fill(d_aos.begin(), d_aos.end(), cuda::std::make_tuple(0, 0));

  thrust::copy(h_soa, h_soa + n, d_aos.begin());
  ASSERT_EQUAL_QUIET(7, cuda::std::get<0>(d_soa[0]));
  ASSERT_EQUAL_QUIET(13, cuda::std::get<1>(d_soa[0]));

  // device to device
  thrust::fill(d_aos.begin(), d_aos.end(), cuda::std::make_tuple(0, 0));

  thrust::copy(d_soa, d_soa + n, d_aos.begin());
  ASSERT_EQUAL_QUIET(7, cuda::std::get<0>(d_soa[0]));
  ASSERT_EQUAL_QUIET(13, cuda::std::get<1>(d_soa[0]));

  // device to host
  thrust::fill(h_aos.begin(), h_aos.end(), cuda::std::make_tuple(0, 0));

  thrust::copy(d_soa, d_soa + n, h_aos.begin());
  ASSERT_EQUAL_QUIET(7, cuda::std::get<0>(h_soa[0]));
  ASSERT_EQUAL_QUIET(13, cuda::std::get<1>(h_soa[0]));
};
DECLARE_UNITTEST(TestZipIteratorCopySoAToAoS);
