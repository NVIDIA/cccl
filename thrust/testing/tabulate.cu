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

TEST_CASE("TestTabulateDispatchExplicit", "[tabulate]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::tabulate(sys, vec.begin(), vec.end(), ::cuda::std::identity{});

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename UnaryOperation>
void tabulate(my_tag, ForwardIterator first, ForwardIterator, UnaryOperation)
{
  *first = 13;
}

TEST_CASE("TestTabulateDispatchImplicit", "[tabulate]")
{
  thrust::device_vector<int> vec(1);

  thrust::tabulate(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), ::cuda::std::identity{});

  REQUIRE(13 == vec.front());
}

template <class Vector>
void test_tabulate_simple()
{
  using namespace thrust::placeholders;

  Vector v(5);

  thrust::tabulate(v.begin(), v.end(), ::cuda::std::identity{});

  Vector ref{0, 1, 2, 3, 4};
  REQUIRE(v == ref);

  thrust::tabulate(v.begin(), v.end(), -_1);

  ref = {0, -1, -2, -3, -4};
  REQUIRE(v == ref);

  thrust::tabulate(v.begin(), v.end(), _1 * _1 * _1);

  ref = {0, 1, 8, 27, 64};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_tabulate_simple);

template <typename T>
void test_tabulate(size_t n)
{
  using namespace thrust::placeholders;

  thrust::host_vector<T> h_data(n);
  thrust::device_vector<T> d_data(n);

  thrust::tabulate(h_data.begin(), h_data.end(), _1 * _1 + 13);
  thrust::tabulate(d_data.begin(), d_data.end(), _1 * _1 + 13);

  REQUIRE(h_data == d_data);

  thrust::tabulate(h_data.begin(), h_data.end(), (_1 - 7) * _1);
  thrust::tabulate(d_data.begin(), d_data.end(), (_1 - 7) * _1);

  REQUIRE(h_data == d_data);
}
DECLARE_VARIABLE_UNITTEST(test_tabulate);

template <typename T>
void test_tabulate_to_discard_iterator(size_t n)
{
  thrust::tabulate(thrust::discard_iterator<thrust::device_system_tag>(),
                   thrust::discard_iterator<thrust::device_system_tag>(static_cast<std::ptrdiff_t>(n)),
                   ::cuda::std::identity{});

  // nothing to check -- just make sure it compiles
}
DECLARE_VARIABLE_UNITTEST(test_tabulate_to_discard_iterator);
