#include <thrust/fill.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/retag.h>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>

#include <unittest/unittest.h>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244 4267) // possible loss of data

template <class Vector>
void test_fill_simple()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  thrust::fill(v.begin() + 1, v.begin() + 4, (T) 7);

  Vector ref{0, 7, 7, 7, 4};
  REQUIRE(v == ref);

  thrust::fill(v.begin() + 0, v.begin() + 3, (T) 8);

  ref = {8, 8, 8, 7, 4};
  REQUIRE(v == ref);

  thrust::fill(v.begin() + 2, v.end(), (T) 9);

  ref = {8, 8, 9, 9, 9};
  REQUIRE(v == ref);

  thrust::fill(v.begin(), v.end(), (T) 1);

  ref = Vector(5, 1);
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_fill_simple);

TEST_CASE("TestFillDiscardIterator", "[fill]")
{
  // there's no result to check because fill returns void
  thrust::fill(
    thrust::discard_iterator<thrust::host_system_tag>(), thrust::discard_iterator<thrust::host_system_tag>(10), 13);

  thrust::fill(
    thrust::discard_iterator<thrust::device_system_tag>(), thrust::discard_iterator<thrust::device_system_tag>(10), 13);
}

template <class Vector>
void test_fill_mixed_types()
{
  Vector v(4);

  thrust::fill(v.begin(), v.end(), bool(true));

  Vector ref(4, 1);
  REQUIRE(v == ref);

  thrust::fill(v.begin(), v.end(), char(20));

  ref = Vector(4, 20);
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_fill_mixed_types);

template <typename T>
void test_fill(size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  thrust::fill(h_data.begin() + std::min((size_t) 1, n), h_data.begin() + std::min((size_t) 3, n), (T) 0);
  thrust::fill(d_data.begin() + std::min((size_t) 1, n), d_data.begin() + std::min((size_t) 3, n), (T) 0);

  REQUIRE(h_data == d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 117, n), h_data.begin() + std::min((size_t) 367, n), (T) 1);
  thrust::fill(d_data.begin() + std::min((size_t) 117, n), d_data.begin() + std::min((size_t) 367, n), (T) 1);

  REQUIRE(h_data == d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 8, n), h_data.begin() + std::min((size_t) 259, n), (T) 2);
  thrust::fill(d_data.begin() + std::min((size_t) 8, n), d_data.begin() + std::min((size_t) 259, n), (T) 2);

  REQUIRE(h_data == d_data);

  thrust::fill(h_data.begin() + std::min((size_t) 3, n), h_data.end(), (T) 3);
  thrust::fill(d_data.begin() + std::min((size_t) 3, n), d_data.end(), (T) 3);

  REQUIRE(h_data == d_data);

  thrust::fill(h_data.begin(), h_data.end(), (T) 4);
  thrust::fill(d_data.begin(), d_data.end(), (T) 4);

  REQUIRE(h_data == d_data);
}
DECLARE_VARIABLE_UNITTEST(test_fill);

template <class Vector>
void test_fill_n_simple()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2, 3, 4};

  typename Vector::iterator iter = thrust::fill_n(v.begin() + 1, 3, (T) 7);

  Vector ref{0, 7, 7, 7, 4};
  REQUIRE(v == ref);

  REQUIRE((v.begin() + 4 == iter));

  iter = thrust::fill_n(v.begin() + 0, 3, (T) 8);

  ref = {8, 8, 8, 7, 4};
  REQUIRE(v == ref);

  REQUIRE((v.begin() + 3 == iter));

  iter = thrust::fill_n(v.begin() + 2, 3, (T) 9);

  ref = {8, 8, 9, 9, 9};
  REQUIRE(v == ref);

  REQUIRE((v.end() == iter));

  iter = thrust::fill_n(v.begin(), v.size(), (T) 1);

  ref = Vector(5, 1);
  REQUIRE(v == ref);

  REQUIRE((v.end() == iter));
}
DECLARE_VECTOR_UNITTEST(test_fill_n_simple);

TEST_CASE("TestFillNDiscardIterator", "[fill]")
{
  const thrust::discard_iterator<thrust::host_system_tag> h_result =
    thrust::fill_n(thrust::discard_iterator<thrust::host_system_tag>(), 10, 13);

  const thrust::discard_iterator<thrust::device_system_tag> d_result =
    thrust::fill_n(thrust::discard_iterator<thrust::device_system_tag>(), 10, 13);

  const thrust::discard_iterator<> reference(10);

  REQUIRE((reference == h_result));
  REQUIRE((reference == d_result));
}

template <class Vector>
void test_fill_n_mixed_types()
{
  Vector v(4);

  typename Vector::iterator iter = thrust::fill_n(v.begin(), v.size(), bool(true));

  Vector ref(4, 1);
  REQUIRE(v == ref);
  REQUIRE((v.end() == iter));

  iter = thrust::fill_n(v.begin(), v.size(), char(20));

  ref = Vector(4, 20);
  REQUIRE(v == ref);
  REQUIRE((v.end() == iter));
}
DECLARE_VECTOR_UNITTEST(test_fill_n_mixed_types);

template <typename T>
void test_fill_n(size_t n)
{
  thrust::host_vector<T> h_data   = unittest::random_integers<T>(n);
  thrust::device_vector<T> d_data = h_data;

  size_t begin_offset = std::min<size_t>(1, n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 3, n) - begin_offset, (T) 0);
  thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t) 3, n) - begin_offset, (T) 0);

  REQUIRE(h_data == d_data);

  begin_offset = std::min<size_t>(117, n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 367, n) - begin_offset, (T) 1);
  thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t) 367, n) - begin_offset, (T) 1);

  REQUIRE(h_data == d_data);

  begin_offset = std::min<size_t>(8, n);
  thrust::fill_n(h_data.begin() + begin_offset, std::min((size_t) 259, n) - begin_offset, (T) 2);
  thrust::fill_n(d_data.begin() + begin_offset, std::min((size_t) 259, n) - begin_offset, (T) 2);

  REQUIRE(h_data == d_data);

  begin_offset = std::min<size_t>(3, n);
  thrust::fill_n(h_data.begin() + begin_offset, h_data.size() - begin_offset, (T) 3);
  thrust::fill_n(d_data.begin() + begin_offset, d_data.size() - begin_offset, (T) 3);

  REQUIRE(h_data == d_data);

  thrust::fill_n(h_data.begin(), h_data.size(), (T) 4);
  thrust::fill_n(d_data.begin(), d_data.size(), (T) 4);

  REQUIRE(h_data == d_data);
}
DECLARE_VARIABLE_UNITTEST(test_fill_n);

template <typename Vector>
void test_fill_zip_iterator()
{
  using T = typename Vector::value_type;

  Vector v1(3, T(0));
  Vector v2(3, T(0));
  Vector v3(3, T(0));

  thrust::fill(thrust::make_zip_iterator(v1.begin(), v2.begin(), v3.begin()),
               thrust::make_zip_iterator(v1.end(), v2.end(), v3.end()),
               cuda::std::tuple<T, T, T>(4, 7, 13));

  Vector ref1{4, 4, 4};
  REQUIRE(ref1 == v1);

  Vector ref2{7, 7, 7};
  REQUIRE(ref2 == v2);

  Vector ref3{13, 13, 13};
  REQUIRE(ref3 == v3);
};
DECLARE_VECTOR_UNITTEST(test_fill_zip_iterator);

TEST_CASE("TestFillTuple", "[fill]")
{
  using T     = int;
  using Tuple = cuda::std::tuple<T, T>;

  thrust::host_vector<Tuple> h(3, Tuple(0, 0));
  thrust::device_vector<Tuple> d(3, Tuple(0, 0));

  thrust::fill(h.begin(), h.end(), Tuple(4, 7));
  thrust::fill(d.begin(), d.end(), Tuple(4, 7));

  REQUIRE((h == d));
}

struct TypeWithTrivialAssigment
{
  int x, y, z;
};

TEST_CASE("TestFillWithTrivialAssignment", "[fill]")
{
  using T = TypeWithTrivialAssigment;

  thrust::host_vector<T> h(1);
  thrust::device_vector<T> d(1);

  REQUIRE(h[0].x == 0);
  REQUIRE(h[0].y == 0);
  REQUIRE(h[0].z == 0);
  REQUIRE(static_cast<T>(d[0]).x == 0);
  REQUIRE(static_cast<T>(d[0]).y == 0);
  REQUIRE(static_cast<T>(d[0]).z == 0);

  T val;
  val.x = 10;
  val.y = 20;
  val.z = -1;

  thrust::fill(h.begin(), h.end(), val);
  thrust::fill(d.begin(), d.end(), val);

  REQUIRE(h[0].x == 10);
  REQUIRE(h[0].y == 20);
  REQUIRE(h[0].z == -1);
  REQUIRE(static_cast<T>(d[0]).x == 10);
  REQUIRE(static_cast<T>(d[0]).y == 20);
  REQUIRE(static_cast<T>(d[0]).z == -1);
}

struct TypeWithNonTrivialAssigment
{
  int x{0}, y{0}, z{0};

  TypeWithNonTrivialAssigment() = default;

  TypeWithNonTrivialAssigment(const TypeWithNonTrivialAssigment&) = default;

  _CCCL_HOST_DEVICE TypeWithNonTrivialAssigment& operator=(const TypeWithNonTrivialAssigment& t)
  {
    x = t.x;
    y = t.y;
    z = t.x + t.y;
    return *this;
  }

  _CCCL_HOST_DEVICE bool operator==(const TypeWithNonTrivialAssigment& t) const
  {
    return x == t.x && y == t.y && z == t.z;
  }
};

TEST_CASE("TestFillWithNonTrivialAssignment", "[fill]")
{
  using T = TypeWithNonTrivialAssigment;

  thrust::host_vector<T> h(1);
  thrust::device_vector<T> d(1);

  REQUIRE(h[0].x == 0);
  REQUIRE(h[0].y == 0);
  REQUIRE(h[0].z == 0);
  REQUIRE(static_cast<T>(d[0]).x == 0);
  REQUIRE(static_cast<T>(d[0]).y == 0);
  REQUIRE(static_cast<T>(d[0]).z == 0);

  T val;
  val.x = 10;
  val.y = 20;
  val.z = -1;

  thrust::fill(h.begin(), h.end(), val);
  thrust::fill(d.begin(), d.end(), val);

  REQUIRE(h[0].x == 10);
  REQUIRE(h[0].y == 20);
  REQUIRE(h[0].z == 30);
  REQUIRE(static_cast<T>(d[0]).x == 10);
  REQUIRE(static_cast<T>(d[0]).y == 20);
  REQUIRE(static_cast<T>(d[0]).z == 30);
}

template <typename ForwardIterator, typename T>
void fill(my_system& system, ForwardIterator /*first*/, ForwardIterator, const T&)
{
  system.validate_dispatch();
}

TEST_CASE("TestFillDispatchExplicit", "[fill]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::fill(sys, vec.begin(), vec.end(), 0);

  REQUIRE(sys.is_valid());
}

template <typename ForwardIterator, typename T>
void fill(my_tag, ForwardIterator first, ForwardIterator, const T&)
{
  *first = 13;
}

TEST_CASE("TestFillDispatchImplicit", "[fill]")
{
  thrust::device_vector<int> vec(1);

  thrust::fill(thrust::retag<my_tag>(vec.begin()), thrust::retag<my_tag>(vec.end()), 0);

  REQUIRE(13 == vec.front());
}

template <typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_system& system, OutputIterator first, Size, const T&)
{
  system.validate_dispatch();
  return first;
}

TEST_CASE("TestFillNDispatchExplicit", "[fill]")
{
  thrust::device_vector<int> vec(1);

  my_system sys(0); // NOLINT(misc-const-correctness)
  thrust::fill_n(sys, vec.begin(), vec.size(), 0);

  REQUIRE(sys.is_valid());
}

template <typename OutputIterator, typename Size, typename T>
OutputIterator fill_n(my_tag, OutputIterator first, Size, const T&)
{
  *first = 13;
  return first;
}

TEST_CASE("TestFillNDispatchImplicit", "[fill]")
{
  thrust::device_vector<int> vec(1);

  thrust::fill_n(thrust::retag<my_tag>(vec.begin()), vec.size(), 0);

  REQUIRE(13 == vec.front());
}

_CCCL_DIAG_POP
