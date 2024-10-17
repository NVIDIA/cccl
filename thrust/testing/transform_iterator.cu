#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <memory>
#include <vector>

#include <unittest/unittest.h>

template <class Vector>
void TestTransformIterator()
{
  using T = typename Vector::value_type;

  using UnaryFunction = thrust::negate<T>;
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

  using UnaryFunction = thrust::negate<T>;
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
    T h_result = thrust::reduce(thrust::make_transform_iterator(h_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(h_data.end(), thrust::negate<T>()));

    // run on device
    T d_result = thrust::reduce(thrust::make_transform_iterator(d_data.begin(), thrust::negate<T>()),
                                thrust::make_transform_iterator(d_data.end(), thrust::negate<T>()));

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

// TODO(bgruber): replace by libc++ with C++14
struct forward
{
  template <class _Tp>
  constexpr _Tp&& operator()(_Tp&& __t) const noexcept
  {
    return _CUDA_VSTD::forward<_Tp>(__t);
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

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, "");
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;
  }

  {
    thrust::device_vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, thrust::device_reference<bool>>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // wrapped reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;
  }

  {
    std::vector<bool> v;

    auto it = v.begin();
    static_assert(is_same<decltype(it)::reference, std::vector<bool>::reference>::value, ""); // proxy reference
    static_assert(is_same<decltype(it)::value_type, bool>::value, "");

    auto it_tr_val = thrust::make_transform_iterator(it, flip_value{});
    static_assert(is_same<decltype(it_tr_val)::reference, bool>::value, "");
    static_assert(is_same<decltype(it_tr_val)::value_type, bool>::value, "");
    (void) it_tr_val;

    auto it_tr_ref = thrust::make_transform_iterator(it, pass_ref{});
    static_assert(is_same<decltype(it_tr_ref)::reference, const bool&>::value, "");
    static_assert(is_same<decltype(it_tr_ref)::value_type, bool>::value, "");
    (void) it_tr_ref;

    auto it_tr_fwd = thrust::make_transform_iterator(it, forward{});
    static_assert(is_same<decltype(it_tr_fwd)::reference, bool&&>::value, ""); // proxy reference is decayed
    static_assert(is_same<decltype(it_tr_fwd)::value_type, bool>::value, "");
    (void) it_tr_fwd;
  }
}
DECLARE_UNITTEST(TestTransformIteratorReferenceAndValueType);
