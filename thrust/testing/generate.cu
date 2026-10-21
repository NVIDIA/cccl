#include <thrust/generate.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <typename T>
struct return_value
{
  T val;

  return_value() = default;
  return_value(T v)
      : val(v)
  {}

  _CCCL_HOST_DEVICE T operator()()
  {
    return val;
  }
};

template <class Vector>
void test_generate_simple()
{
  using T = typename Vector::value_type;

  Vector result(5);

  const T value = 13;

  const return_value<T> f(value);

  thrust::generate(result.begin(), result.end(), f);

  Vector ref(result.size(), value);
  REQUIRE(result == ref);
}
DECLARE_VECTOR_UNITTEST(test_generate_simple);

template <typename ForwardIterator, typename Generator>
void generate(my_system& system, ForwardIterator /*first*/, ForwardIterator, Generator)
{
  system.validate_dispatch();
}

TEST_CASE("TestGenerateDispatchExplicit", "[generate]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::generate(sys, vec.begin(), vec.end(), 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename Generator>
void generate(my_tag, ForwardIterator first, ForwardIterator, Generator)
{
  *first = 13;
}

TEST_CASE("TestGenerateDispatchImplicit", "[generate]")
{
  thrust::device_vector<int> vec(1);

  thrust::generate(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  REQUIRE(13 == vec.front());
}

template <typename T>
void test_generate(const size_t n)
{
  thrust::host_vector<T> h_result(n);
  thrust::device_vector<T> d_result(n);

  T value = 13;
  const return_value<T> f(value);

  thrust::generate(h_result.begin(), h_result.end(), f);
  thrust::generate(d_result.begin(), d_result.end(), f);

  REQUIRE(h_result == d_result);
}
DECLARE_VARIABLE_UNITTEST(test_generate);

template <typename T>
void test_generate_to_discard_iterator(const size_t)
{
  T value = 13;
  const return_value<T> f(value);

  thrust::discard_iterator<thrust::host_system_tag> h_first; // NOLINT(misc-const-correctness)
  thrust::generate(h_first, h_first + 10, f);

  thrust::discard_iterator<thrust::device_system_tag> d_first; // NOLINT(misc-const-correctness)
  thrust::generate(d_first, d_first + 10, f);

  // there's nothing to actually check except that it compiles
}
DECLARE_VARIABLE_UNITTEST(test_generate_to_discard_iterator);

template <class Vector>
void test_generate_n_simple()
{
  using T = typename Vector::value_type;

  Vector result(5);

  const T value = 13;

  const return_value<T> f(value);

  thrust::generate_n(result.begin(), result.size(), f);

  Vector ref(result.size(), value);
  REQUIRE(result == ref);
}
DECLARE_VECTOR_UNITTEST(test_generate_n_simple);

template <typename ForwardIterator, typename Size, typename Generator>
ForwardIterator generate_n(my_system& system, ForwardIterator first, Size, Generator)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("TestGenerateNDispatchExplicit", "[generate]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::generate_n(sys, vec.begin(), vec.size(), 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename Size, typename Generator>
ForwardIterator generate_n(my_tag, ForwardIterator first, Size, Generator)
{
  *first = 13;
  return first;
}

TEST_CASE("TestGenerateNDispatchImplicit", "[generate]")
{
  thrust::device_vector<int> vec(1);

  thrust::generate_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

  REQUIRE(13 == vec.front());
}

template <typename T>
void test_generate_n_to_discard_iterator(const size_t n)
{
  T value = 13;
  const return_value<T> f(value);

  const thrust::discard_iterator<thrust::host_system_tag> h_result =
    thrust::generate_n(thrust::discard_iterator<thrust::host_system_tag>(), n, f);

  const thrust::discard_iterator<thrust::device_system_tag> d_result =
    thrust::generate_n(thrust::discard_iterator<thrust::device_system_tag>(), n, f);

  const thrust::discard_iterator<> reference(static_cast<std::ptrdiff_t>(n));

  REQUIRE((reference == h_result));
  REQUIRE((reference == d_result));
}
DECLARE_VARIABLE_UNITTEST(test_generate_n_to_discard_iterator);

template <typename Vector>
void test_generate_zip_iterator()
{
  using T = typename Vector::value_type;

  Vector v1(3, T(0));
  Vector v2(3, T(0));

  thrust::generate(thrust::make_zip_iterator(v1.begin(), v2.begin()),
                   thrust::make_zip_iterator(v1.end(), v2.end()),
                   return_value<cuda::std::tuple<T, T>>(cuda::std::tuple<T, T>(4, 7)));

  Vector ref1(3, 4);
  Vector ref2(3, 7);
  REQUIRE(v1 == ref1);
  REQUIRE(v2 == ref2);
};
DECLARE_VECTOR_UNITTEST(test_generate_zip_iterator);

TEST_CASE("TestGenerateTuple", "[generate]")
{
  using T     = int;
  using Tuple = cuda::std::tuple<T, T>;

  thrust::host_vector<Tuple> h(3, Tuple(0, 0));
  thrust::device_vector<Tuple> d(3, Tuple(0, 0));

  thrust::generate(h.begin(), h.end(), return_value<Tuple>(Tuple(4, 7)));
  thrust::generate(d.begin(), d.end(), return_value<Tuple>(Tuple(4, 7)));

  REQUIRE((h == d));
}

_CCCL_DIAG_POP
