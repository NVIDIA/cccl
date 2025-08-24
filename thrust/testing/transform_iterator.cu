#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <memory>
#include <vector>

#include <unittest/unittest.h>

// ensure that we properly support thrust::transform_iterator from cuda::std
void TestTransformIteratorTraits()
{
  using func    = ::cuda::std::negate<int>;
  using base_it = thrust::host_vector<int>::iterator;

  using it     = thrust::transform_iterator<func, base_it>;
  using traits = cuda::std::iterator_traits<it>;

  static_assert(cuda::std::is_same_v<traits::difference_type, ptrdiff_t>);
  static_assert(cuda::std::is_same_v<traits::value_type, int>);
  static_assert(cuda::std::is_same_v<traits::pointer, void>);
  static_assert(cuda::std::is_same_v<traits::reference, int>);
  static_assert(cuda::std::is_same_v<traits::iterator_category, ::cuda::std::random_access_iterator_tag>);

  static_assert(cuda::std::is_same_v<thrust::iterator_traversal_t<it>, thrust::random_access_traversal_tag>);

  static_assert(cuda::std::__is_cpp17_random_access_iterator<it>::value);

  static_assert(!cuda::std::output_iterator<it, int>);
  static_assert(cuda::std::input_iterator<it>);
  static_assert(cuda::std::forward_iterator<it>);
  static_assert(cuda::std::bidirectional_iterator<it>);
  static_assert(cuda::std::random_access_iterator<it>);
  static_assert(!cuda::std::contiguous_iterator<it>);
}
DECLARE_UNITTEST(TestTransformIteratorTraits);

template <class Vector>
void TestTransformIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = ::cuda::std::negate<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

  thrust::copy(iter, iter + 4, output.begin());

  Vector ref{-1, -2, -3, -4};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestTransformIterator);

template <class Vector>
void TestMakeTransformIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = ::cuda::std::negate<T>;
  using Iterator      = typename Vector::iterator;

  Vector input(4);
  Vector output(4);

  // initialize input
  thrust::sequence(input.begin(), input.end(), 1);

  // construct transform_iterator
  thrust::transform_iterator<UnaryFunction, Iterator> iter(input.begin(), UnaryFunction());

  thrust::copy(thrust::make_transform_iterator(input.begin(), UnaryFunction()),
               thrust::make_transform_iterator(input.end(), UnaryFunction()),
               output.begin());

  Vector ref{-1, -2, -3, -4};
  ASSERT_EQUAL(output, ref);
}
DECLARE_VECTOR_UNITTEST(TestMakeTransformIterator);

template <typename T>
struct TestTransformIteratorReduce
{
  void operator()(const size_t n)
  {
    thrust::host_vector<T> h_data   = unittest::random_samples<T>(n);
    thrust::device_vector<T> d_data = h_data;

    // run on host
    T h_result = thrust::reduce(thrust::make_transform_iterator(h_data.begin(), ::cuda::std::negate<T>()),
                                thrust::make_transform_iterator(h_data.end(), ::cuda::std::negate<T>()));

    // run on device
    T d_result = thrust::reduce(thrust::make_transform_iterator(d_data.begin(), ::cuda::std::negate<T>()),
                                thrust::make_transform_iterator(d_data.end(), ::cuda::std::negate<T>()));

    ASSERT_EQUAL(h_result, d_result);
  }
};
VariableUnitTest<TestTransformIteratorReduce, IntegralTypes> TestTransformIteratorReduceInstance;

struct ExtractValue
{
  int operator()(std::unique_ptr<int> const& n)
  {
    return *n;
  }
};

void TestTransformIteratorNonCopyable()
{
  thrust::host_vector<std::unique_ptr<int>> hv(4);
  hv[0].reset(new int{1});
  hv[1].reset(new int{2});
  hv[2].reset(new int{3});
  hv[3].reset(new int{4});

  auto transformed = thrust::make_transform_iterator(hv.begin(), ExtractValue{});
  ASSERT_EQUAL(transformed[0], 1);
  ASSERT_EQUAL(transformed[1], 2);
  ASSERT_EQUAL(transformed[2], 3);
  ASSERT_EQUAL(transformed[3], 4);
}

DECLARE_UNITTEST(TestTransformIteratorNonCopyable);

struct flip_value
{
  _CCCL_HOST_DEVICE bool operator()(bool b) const
  {
    return !b;
  }
};

struct pass_ref
{
  _CCCL_HOST_DEVICE const bool& operator()(const bool& b) const
  {
    return b;
  }
};

// a user provided functor that forwards its argument
struct forward
{
  template <class _Tp>
  constexpr _Tp&& operator()(_Tp&& __t) const noexcept
  {
    return ::cuda::std::forward<_Tp>(__t);
  }
};

void TestTransformIteratorReferenceAndValueType()
{
  using ::cuda::std::is_same;
  using ::cuda::std::negate;
  {
    thrust::host_vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, bool&>::value, ""); // ordinary reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, "");
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_cid = thrust::make_transform_iterator(it, cuda::std::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
  }

  {
    thrust::device_vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, thrust::device_reference<bool>>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // wrapped reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_cid = thrust::make_transform_iterator(it, cuda::std::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
  }

  {
    std::vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, std::vector<bool>::reference>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // proxy reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");

    [[maybe_unused]] auto it_tr_cid = thrust::make_transform_iterator(it, cuda::std::identity{});
    static_assert(is_same<decltype(it_tr_cid)::reference, bool>::value, ""); // special handling by
                                                                             // transform_iterator_reference
    static_assert(is_same<decltype(it_tr_cid)::value_type, bool>::value, "");
  }
}
DECLARE_UNITTEST(TestTransformIteratorReferenceAndValueType);

void TestTransformIteratorIdentity()
{
  thrust::device_vector<int> v(3, 42);

  ASSERT_EQUAL(*thrust::make_transform_iterator(v.begin(), cuda::std::identity{}), 42);
  using namespace thrust::placeholders;
  ASSERT_EQUAL(*thrust::make_transform_iterator(v.begin(), _1), 42);
}

DECLARE_UNITTEST(TestTransformIteratorIdentity);
