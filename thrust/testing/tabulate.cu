#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/tabulate.h>

#include <unittest/unittest.h>

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_system& system, ForwardIterator, ForwardIterator, UnaryOperation)
{
  system.validate_dispatch();
}

void TestTabulateDispatchExplicit()
{
  thrust::device_vector<int> vec(1);

  my_system sys(0);
  thrust::tabulate(sys, vec.begin(), vec.end(), thrust::identity<int>());

  ASSERT_EQUAL(true, sys.is_valid());
}
DECLARE_UNITTEST(TestTabulateDispatchExplicit);

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_tag, ForwardIterator first, ForwardIterator, UnaryOperation)
{
  *first = 13;
}

void TestTabulateDispatchImplicit()
{
  thrust::device_vector<int> vec(1);

  thrust::tabulate(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), thrust::identity<int>());

  ASSERT_EQUAL(13, vec.front());
}
DECLARE_UNITTEST(TestTabulateDispatchImplicit);

template <class Vector>
void TestTabulateSimple()
{
  using namespace thrust::placeholders;
  using T = typename Vector::value_type;

  Vector v(5);

  thrust::tabulate(v.begin(), v.end(), thrust::identity<T>());

  Vector ref{0, 1, 2, 3, 4};
  ASSERT_EQUAL(v, ref);

  thrust::tabulate(v.begin(), v.end(), -_1);

  ref = {0, -1, -2, -3, -4};
  ASSERT_EQUAL(v, ref);

  thrust::tabulate(v.begin(), v.end(), _1 * _1 * _1);

  ref = {0, 1, 8, 27, 64};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestTabulateSimple);

template <typename T>
void TestTabulate(size_t n)
{
  using namespace thrust::placeholders;

  thrust::host_vector<T> h_data(n);
  thrust::device_vector<T> d_data(n);

  thrust::tabulate(h_data.begin(), h_data.end(), _1 * _1 + 13);
  thrust::tabulate(d_data.begin(), d_data.end(), _1 * _1 + 13);

  ASSERT_EQUAL(h_data, d_data);

  thrust::tabulate(h_data.begin(), h_data.end(), (_1 - 7) * _1);
  thrust::tabulate(d_data.begin(), d_data.end(), (_1 - 7) * _1);

  ASSERT_EQUAL(h_data, d_data);
}
DECLARE_VARIABLE_UNITTEST(TestTabulate);

template <typename T>
void TestTabulateToDiscardIterator(size_t n)
{
  thrust::tabulate(thrust::discard_iterator<thrust::device_system_tag>(),
                   thrust::discard_iterator<thrust::device_system_tag>(n),
                   thrust::identity<int>());

  // nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(TestTabulateToDiscardIterator);
