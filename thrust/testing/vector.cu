#include <thrust/detail/config.h>

// gcc >= 11 emits bogus -Werror=stringop-overflow and -Werror=array-bounds diagnostics for the memmove that thrust uses
// to copy small vectors of narrow types (e.g. host_vector<signed char>). This needs to be suppressed before any header
// pulls in the memmove implementation, since gcc ties the diagnostic state to where that code is first parsed.
_CCCL_DIAG_SUPPRESS_GCC("-Wstringop-overflow")
_CCCL_DIAG_SUPPRESS_GCC("-Warray-bounds")

#include <thrust/count.h>
#include <thrust/device_malloc_allocator.h>
#include <thrust/sequence.h>

#include <initializer_list>
#include <limits>
#include <list>
#include <utility>
#include <vector>

#include <unittest/unittest.h>

template <class Vector>
void test_vector_zero_size()
{
  Vector v;
  REQUIRE(v.size() == 0lu);
  REQUIRE(v.begin() == v.end());
}
DECLARE_VECTOR_UNITTEST(test_vector_zero_size);

TEST_CASE("TestVectorBool", "[vector]")
{
  const thrust::host_vector<bool> h{true, false, true};
  const thrust::device_vector<bool> d{true, false, true};

  const thrust::host_vector<bool> h_ref{true, false, true};
  const thrust::device_vector<bool> d_ref{true, false, true};
  REQUIRE(h == h_ref);
  REQUIRE(d == d_ref);
}

template <class Vector>
void test_vector_initializer_list()
{
  Vector v{1, 2, 3};
  REQUIRE(v.size() == 3lu);
  Vector ref{1, 2, 3};
  REQUIRE(v == ref);

  v = {1, 2, 3, 4};
  REQUIRE(v.size() == 4lu);
  Vector v_ref = {1, 2, 3, 4};
  REQUIRE(v == v_ref);

  const auto alloc = v.get_allocator();
  Vector v2{{1, 2, 3}, alloc};
  REQUIRE(v2.size() == 3lu);
  Vector v2_ref = {1, 2, 3};
  REQUIRE(v2 == v2_ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_initializer_list);

template <class Vector>
void test_vector_front_back()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2};

  REQUIRE(v.front() == T(0));
  REQUIRE(v.back() == T(2));
}
DECLARE_VECTOR_UNITTEST(test_vector_front_back);

template <class Vector>
void test_vector_data()
{
  using PointerT      = typename Vector::pointer;
  using PointerConstT = typename Vector::const_pointer;

  Vector v{0, 1, 2};

  REQUIRE(0 == *v.data());
  REQUIRE(1 == *(v.data() + 1));
  REQUIRE(2 == *(v.data() + 2));
  REQUIRE(PointerT(&v.front()) == v.data());
  REQUIRE(PointerT(&*v.begin()) == v.data());
  REQUIRE(PointerT(&v[0]) == v.data());

  const Vector& c_v = v;

  REQUIRE(0 == *c_v.data());
  REQUIRE(1 == *(c_v.data() + 1));
  REQUIRE(2 == *(c_v.data() + 2));
  REQUIRE(PointerConstT(&c_v.front()) == c_v.data());
  REQUIRE(PointerConstT(&*c_v.begin()) == c_v.data());
  REQUIRE(PointerConstT(&c_v[0]) == c_v.data());
}
DECLARE_VECTOR_UNITTEST(test_vector_data);

template <class Vector>
void test_vector_element_assignment()
{
  Vector v{0, 1, 2};

  Vector ref{0, 1, 2};
  REQUIRE(v == ref);

  v   = {10, 11, 12};
  ref = {10, 11, 12};
  REQUIRE(v == ref);

  Vector w = v;
  REQUIRE(v == w);
}
DECLARE_VECTOR_UNITTEST(test_vector_element_assignment);

template <class Vector>
void test_vector_from_stl_vector()
{
  using T = typename Vector::value_type;

  const std::vector<T> stl_vector{0, 1, 2};

  thrust::host_vector<T> v(stl_vector);

  REQUIRE(v.size() == 3lu);
  const thrust::host_vector<T> ref{0, 1, 2};
  REQUIRE(v == ref);

  v = stl_vector;

  REQUIRE(v.size() == 3lu);
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_from_stl_vector);

template <class Vector>
void test_vector_fill_assign()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> v;
  v.assign(3, 13);

  REQUIRE(v.size() == 3lu);
  const thrust::host_vector<T> ref{13, 13, 13};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_fill_assign);

template <class Vector>
THRUST_DISABLE_BROKEN_GCC_VECTORIZER void test_vector_fill_insert()
{
  { // Insert into empty vector
    Vector v;
    v.insert(v.end(), 3, 13);

    REQUIRE(v.size() == 3lu);
    Vector ref{13, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector at end
    Vector v{13, 13, 13};
    v.insert(v.end(), 2, 42);

    REQUIRE(v.size() == 5lu);
    Vector ref{13, 13, 13, 42, 42};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector at front, existing elements inserted before end
    Vector v{13, 13, 13};
    v.insert(v.begin(), 2, 42);

    REQUIRE(v.size() == 5lu);
    Vector ref{42, 42, 13, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector at front, existing elements inserted after end
    Vector v{13, 13, 13};
    v.insert(v.begin(), 4, 42);

    REQUIRE(v.size() == 7lu);
    Vector ref{42, 42, 42, 42, 13, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted before end
    Vector v{13, 13, 13};
    v.insert(v.begin() + 1, 1, 42);

    REQUIRE(v.size() == 4lu);
    Vector ref{13, 42, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted at end
    Vector v{13, 13, 13};
    v.insert(v.begin() + 1, 2, 42);

    REQUIRE(v.size() == 5lu);
    Vector ref{13, 42, 42, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted after end
    Vector v{13, 13, 13};
    v.insert(v.begin() + 1, 4, 42);

    REQUIRE(v.size() == 7lu);
    Vector ref{13, 42, 42, 42, 42, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into empty vector, with sufficient capacity
    Vector v;
    v.reserve(42);
    v.insert(v.end(), 3, 13);

    REQUIRE(v.size() == 3lu);
    Vector ref{13, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector at end, with sufficient capacity
    Vector v{13, 13, 13};
    v.reserve(42);
    v.insert(v.end(), 2, 42);

    REQUIRE(v.size() == 5lu);
    Vector ref{13, 13, 13, 42, 42};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector at front, existing elements inserted after end, with sufficient capacity
    Vector v{13, 13, 13};
    v.reserve(42);
    v.insert(v.begin(), 4, 42);

    REQUIRE(v.size() == 7lu);
    Vector ref{42, 42, 42, 42, 13, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted before end, with sufficient capacity
    Vector v{13, 13, 13};
    v.reserve(42);
    v.insert(v.begin() + 1, 1, 42);

    REQUIRE(v.size() == 4lu);
    Vector ref{13, 42, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted at end, with sufficient capacity
    Vector v{13, 13, 13};
    v.reserve(42);
    v.insert(v.begin() + 1, 2, 42);

    REQUIRE(v.size() == 5lu);
    Vector ref{13, 42, 42, 13, 13};
    REQUIRE(v == ref);
  }

  { // Insert into non-empty vector in middle, existing elements inserted after end, with sufficient capacity
    Vector v{13, 13, 13};
    v.reserve(42);
    v.insert(v.begin() + 1, 4, 42);

    REQUIRE(v.size() == 7lu);
    Vector ref{13, 42, 42, 42, 42, 13, 13};
    REQUIRE(v == ref);
  }
}
DECLARE_VECTOR_UNITTEST(test_vector_fill_insert);

template <class Vector>
void test_vector_assign_from_stl_vector()
{
  using T = typename Vector::value_type;

  std::vector<T> stl_vector{0, 1, 2};

  thrust::host_vector<T> v;
  v.assign(stl_vector.begin(), stl_vector.end());

  REQUIRE(v.size() == 3lu);
  const thrust::host_vector<T> ref{0, 1, 2};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_assign_from_stl_vector);

template <class Vector>
void test_vector_from_bi_directional_iterator()
{
  using T = typename Vector::value_type;

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v(stl_list.begin(), stl_list.end());

  REQUIRE(v.size() == 3lu);
  Vector ref{0, 1, 2};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_from_bi_directional_iterator);

template <class Vector>
void test_vector_assign_from_bi_directional_iterator()
{
  using T = typename Vector::value_type;

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v;
  v.assign(stl_list.begin(), stl_list.end());

  REQUIRE(v.size() == 3lu);
  Vector ref{0, 1, 2};
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_assign_from_bi_directional_iterator);

template <class Vector>
void test_vector_assign_from_host_vector()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> h{0, 1, 2};

  Vector v;
  v.assign(h.begin(), h.end());

  REQUIRE(v == h);
}
DECLARE_VECTOR_UNITTEST(test_vector_assign_from_host_vector);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wself-assign")

template <class Vector>
void test_vector_to_and_from_host_vector()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> h{0, 1, 2};

  Vector v(h);

  REQUIRE(v == h);

  // NOLINTNEXTLINE(misc-redundant-expression): self-assignment is what this test checks
  v = v;

  REQUIRE(v == h);

  v = {10, 11, 12};
  Vector v_ref{10, 11, 12};
  REQUIRE(v == v_ref);

  Vector h_ref{0, 1, 2};
  REQUIRE(h == h_ref);

  h = v;

  REQUIRE(v == h);

  h[1] = 11;

  v = h;

  REQUIRE(v == h);
}
DECLARE_VECTOR_UNITTEST(test_vector_to_and_from_host_vector);

_CCCL_DIAG_POP

template <class Vector>
void test_vector_assign_from_device_vector()
{
  using T = typename Vector::value_type;

  thrust::device_vector<T> d{0, 1, 2};

  Vector v;
  v.assign(d.begin(), d.end());

  REQUIRE(v == d);
}
DECLARE_VECTOR_UNITTEST(test_vector_assign_from_device_vector);

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wself-assign")

template <class Vector>
void test_vector_to_and_from_device_vector()
{
  using T = typename Vector::value_type;

  thrust::device_vector<T> h{0, 1, 2};

  Vector v(h);

  REQUIRE(v == h);

  // NOLINTNEXTLINE(misc-redundant-expression): self-assignment is what this test checks
  v = v;

  REQUIRE(v == h);

  v = {10, 11, 12};
  Vector v_ref{10, 11, 12};
  REQUIRE(v == v_ref);

  Vector h_ref{0, 1, 2};
  REQUIRE(h == h_ref);

  h = v;

  REQUIRE(v == h);

  h[1] = 11;

  v = h;

  REQUIRE(v == h);
}
DECLARE_VECTOR_UNITTEST(test_vector_to_and_from_device_vector);
_CCCL_DIAG_POP

template <class Vector>
void test_vector_with_initial_value()
{
  using T = typename Vector::value_type;

  const T init = 17;

  Vector v(3, init);

  REQUIRE(v.size() == 3lu);
  Vector ref(3, init);
  REQUIRE(v == ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_with_initial_value);

template <class Vector>
void test_vector_swap()
{
  Vector v{0, 1, 2};
  Vector u{10, 11, 12};

  v.swap(u);

  Vector u_ref{0, 1, 2};
  REQUIRE(u == u_ref);

  Vector v_ref{10, 11, 12};
  REQUIRE(v == v_ref);
}
DECLARE_VECTOR_UNITTEST(test_vector_swap);

template <class Vector>
void test_vector_erase_position()
{
  Vector v{0, 1, 2, 3, 4};

  v.erase(v.begin() + 2);

  REQUIRE(v.size() == 4lu);
  Vector ref{0, 1, 3, 4};
  REQUIRE(v == ref);

  v.erase(v.begin() + 0);

  REQUIRE(v.size() == 3lu);
  ref = {1, 3, 4};
  REQUIRE(v == ref);

  v.erase(v.begin() + 2);

  REQUIRE(v.size() == 2lu);
  ref = {1, 3};
  REQUIRE(v == ref);

  v.erase(v.begin() + 1);

  REQUIRE(v.size() == 1lu);
  REQUIRE(v[0] == 1);

  v.erase(v.begin() + 0);

  REQUIRE(v.size() == 0lu);
}
DECLARE_VECTOR_UNITTEST(test_vector_erase_position);

template <class Vector>
void test_vector_erase_range()
{
  Vector v{0, 1, 2, 3, 4, 5};

  v.erase(v.begin() + 1, v.begin() + 3);

  REQUIRE(v.size() == 4lu);
  Vector ref{0, 3, 4, 5};
  REQUIRE(v == ref);

  v.erase(v.begin() + 2, v.end());

  REQUIRE(v.size() == 2lu);
  ref = {0, 3};
  REQUIRE(v == ref);

  v.erase(v.begin() + 0, v.begin() + 1);

  REQUIRE(v.size() == 1lu);
  REQUIRE(v[0] == 3);

  v.erase(v.begin(), v.end());

  REQUIRE(v.size() == 0lu);
}
DECLARE_VECTOR_UNITTEST(test_vector_erase_range);

TEST_CASE("TestVectorEquality", "[vector]")
{
  const thrust::host_vector<int> h_a{0, 1, 2};
  const thrust::host_vector<int> h_b{0, 1, 3};
  const thrust::host_vector<int> h_c(3);

  const thrust::device_vector<int> d_a{0, 1, 2};
  const thrust::device_vector<int> d_b{0, 1, 3};
  const thrust::device_vector<int> d_c(3);

  const std::vector<int> s_a{0, 1, 2};
  const std::vector<int> s_b{0, 1, 3};
  const std::vector<int> s_c(3);

  REQUIRE(h_a == h_a);
  REQUIRE(h_a == d_a);
  REQUIRE(d_a == h_a);
  REQUIRE(d_a == d_a);
  REQUIRE(h_b == h_b);
  REQUIRE(h_b == d_b);
  REQUIRE(d_b == h_b);
  REQUIRE(d_b == d_b);
  REQUIRE(h_c == h_c);
  REQUIRE(h_c == d_c);
  REQUIRE(d_c == h_c);
  REQUIRE(d_c == d_c);

  // test vector vs device_vector
  REQUIRE(s_a == d_a);
  REQUIRE(d_a == s_a);
  REQUIRE(s_b == d_b);
  REQUIRE(d_b == s_b);
  REQUIRE(s_c == d_c);
  REQUIRE(d_c == s_c);

  // test vector vs host_vector
  REQUIRE(s_a == h_a);
  REQUIRE(h_a == s_a);
  REQUIRE(s_b == h_b);
  REQUIRE(h_b == s_b);
  REQUIRE(s_c == h_c);
  REQUIRE(h_c == s_c);

  REQUIRE_FALSE(h_a == h_b);
  REQUIRE_FALSE(h_a == d_b);
  REQUIRE_FALSE(d_a == h_b);
  REQUIRE_FALSE(d_a == d_b);
  REQUIRE_FALSE(h_b == h_a);
  REQUIRE_FALSE(h_b == d_a);
  REQUIRE_FALSE(d_b == h_a);
  REQUIRE_FALSE(d_b == d_a);
  REQUIRE_FALSE(h_a == h_c);
  REQUIRE_FALSE(h_a == d_c);
  REQUIRE_FALSE(d_a == h_c);
  REQUIRE_FALSE(d_a == d_c);
  REQUIRE_FALSE(h_c == h_a);
  REQUIRE_FALSE(h_c == d_a);
  REQUIRE_FALSE(d_c == h_a);
  REQUIRE_FALSE(d_c == d_a);
  REQUIRE_FALSE(h_b == h_c);
  REQUIRE_FALSE(h_b == d_c);
  REQUIRE_FALSE(d_b == h_c);
  REQUIRE_FALSE(d_b == d_c);
  REQUIRE_FALSE(h_c == h_b);
  REQUIRE_FALSE(h_c == d_b);
  REQUIRE_FALSE(d_c == h_b);
  REQUIRE_FALSE(d_c == d_b);

  // test vector vs device_vector
  REQUIRE_FALSE(s_a == d_b);
  REQUIRE_FALSE(d_a == s_b);
  REQUIRE_FALSE(s_b == d_a);
  REQUIRE_FALSE(d_b == s_a);
  REQUIRE_FALSE(s_a == d_c);
  REQUIRE_FALSE(d_a == s_c);
  REQUIRE_FALSE(s_c == d_a);
  REQUIRE_FALSE(d_c == s_a);
  REQUIRE_FALSE(s_b == d_c);
  REQUIRE_FALSE(d_b == s_c);
  REQUIRE_FALSE(s_c == d_b);
  REQUIRE_FALSE(d_c == s_b);

  // test vector vs host_vector
  REQUIRE_FALSE(s_a == h_b);
  REQUIRE_FALSE(h_a == s_b);
  REQUIRE_FALSE(s_b == h_a);
  REQUIRE_FALSE(h_b == s_a);
  REQUIRE_FALSE(s_a == h_c);
  REQUIRE_FALSE(h_a == s_c);
  REQUIRE_FALSE(s_c == h_a);
  REQUIRE_FALSE(h_c == s_a);
  REQUIRE_FALSE(s_b == h_c);
  REQUIRE_FALSE(h_b == s_c);
  REQUIRE_FALSE(s_c == h_b);
  REQUIRE_FALSE(h_c == s_b);
}

TEST_CASE("TestVectorInequality", "[vector]")
{
  const thrust::host_vector<int> h_a{0, 1, 2};
  const thrust::host_vector<int> h_b{0, 1, 3};
  const thrust::host_vector<int> h_c(3);

  const thrust::device_vector<int> d_a{0, 1, 2};
  const thrust::device_vector<int> d_b{0, 1, 3};
  const thrust::device_vector<int> d_c(3);

  const std::vector<int> s_a{0, 1, 2};
  const std::vector<int> s_b{0, 1, 3};
  const std::vector<int> s_c(3);

  REQUIRE_FALSE(h_a != h_a);
  REQUIRE_FALSE(h_a != d_a);
  REQUIRE_FALSE(d_a != h_a);
  REQUIRE_FALSE(d_a != d_a);
  REQUIRE_FALSE(h_b != h_b);
  REQUIRE_FALSE(h_b != d_b);
  REQUIRE_FALSE(d_b != h_b);
  REQUIRE_FALSE(d_b != d_b);
  REQUIRE_FALSE(h_c != h_c);
  REQUIRE_FALSE(h_c != d_c);
  REQUIRE_FALSE(d_c != h_c);
  REQUIRE_FALSE(d_c != d_c);

  // test vector vs device_vector
  REQUIRE_FALSE(s_a != d_a);
  REQUIRE_FALSE(d_a != s_a);
  REQUIRE_FALSE(s_b != d_b);
  REQUIRE_FALSE(d_b != s_b);
  REQUIRE_FALSE(s_c != d_c);
  REQUIRE_FALSE(d_c != s_c);

  // test vector vs host_vector
  REQUIRE_FALSE(s_a != h_a);
  REQUIRE_FALSE(h_a != s_a);
  REQUIRE_FALSE(s_b != h_b);
  REQUIRE_FALSE(h_b != s_b);
  REQUIRE_FALSE(s_c != h_c);
  REQUIRE_FALSE(h_c != s_c);

  REQUIRE(h_a != h_b);
  REQUIRE(h_a != d_b);
  REQUIRE(d_a != h_b);
  REQUIRE(d_a != d_b);
  REQUIRE(h_b != h_a);
  REQUIRE(h_b != d_a);
  REQUIRE(d_b != h_a);
  REQUIRE(d_b != d_a);
  REQUIRE(h_a != h_c);
  REQUIRE(h_a != d_c);
  REQUIRE(d_a != h_c);
  REQUIRE(d_a != d_c);
  REQUIRE(h_c != h_a);
  REQUIRE(h_c != d_a);
  REQUIRE(d_c != h_a);
  REQUIRE(d_c != d_a);
  REQUIRE(h_b != h_c);
  REQUIRE(h_b != d_c);
  REQUIRE(d_b != h_c);
  REQUIRE(d_b != d_c);
  REQUIRE(h_c != h_b);
  REQUIRE(h_c != d_b);
  REQUIRE(d_c != h_b);
  REQUIRE(d_c != d_b);

  // test vector vs device_vector
  REQUIRE(s_a != d_b);
  REQUIRE(d_a != s_b);
  REQUIRE(s_b != d_a);
  REQUIRE(d_b != s_a);
  REQUIRE(s_a != d_c);
  REQUIRE(d_a != s_c);
  REQUIRE(s_c != d_a);
  REQUIRE(d_c != s_a);
  REQUIRE(s_b != d_c);
  REQUIRE(d_b != s_c);
  REQUIRE(s_c != d_b);
  REQUIRE(d_c != s_b);

  // test vector vs host_vector
  REQUIRE(s_a != h_b);
  REQUIRE(h_a != s_b);
  REQUIRE(s_b != h_a);
  REQUIRE(h_b != s_a);
  REQUIRE(s_a != h_c);
  REQUIRE(h_a != s_c);
  REQUIRE(s_c != h_a);
  REQUIRE(h_c != s_a);
  REQUIRE(s_b != h_c);
  REQUIRE(h_b != s_c);
  REQUIRE(s_c != h_b);
  REQUIRE(h_c != s_b);
}

template <class Vector>
void test_vector_resizing()
{
  Vector v;

  v.resize(3);

  REQUIRE(v.size() == 3lu);

  v = {0, 1, 2};
  v.resize(5);

  REQUIRE(v.size() == 5lu);

  Vector ref{0, 1, 2, v[3], v[4]};
  REQUIRE(v == ref);

  v[3] = 3;
  v[4] = 4;

  v.resize(4);

  REQUIRE(v.size() == 4lu);

  ref = {0, 1, 2, 3};
  REQUIRE(v == ref);

  v.resize(0);

  REQUIRE(v.size() == 0lu);
}
DECLARE_VECTOR_UNITTEST(test_vector_resizing);

template <class Vector>
void test_vector_reserving()
{
  Vector v;

  v.reserve(3);

  REQUIRE(v.capacity() >= 3lu);

  const size_t old_capacity = v.capacity();

  v.reserve(0);

  REQUIRE(v.capacity() == old_capacity);
}
DECLARE_VECTOR_UNITTEST(test_vector_reserving)

template <class Vector>
void test_vector_uninitialised_copy()
{
  thrust::device_vector<int> v;
  const std::vector<int> std_vector;

  v = std_vector;

  REQUIRE(v.size() == static_cast<size_t>(0));
}
DECLARE_VECTOR_UNITTEST(test_vector_uninitialised_copy);

template <class Vector>
void test_vector_shrink_to_fit()
{
  using T = typename Vector::value_type;

  Vector v;

  v.reserve(200);

  REQUIRE(v.capacity() >= 200lu);

  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  v.shrink_to_fit();

  REQUIRE(T(1) == v[0]);
  REQUIRE(T(2) == v[1]);
  REQUIRE(T(3) == v[2]);
  REQUIRE(3lu == v.size());
  REQUIRE(3lu == v.capacity());
}
DECLARE_VECTOR_UNITTEST(test_vector_shrink_to_fit)

template <int N>
struct LargeStruct
{
  int data[N];

  _CCCL_HOST_DEVICE bool operator==(const LargeStruct& ls) const
  {
    for (int i = 0; i < N; i++)
    {
      if (data[i] != ls.data[i])
      {
        return false;
      }
    }
    return true;
  }
};

TEST_CASE("TestVectorContainingLargeType", "[vector]")
{
  // Thrust issue #5
  // http://code.google.com/p/thrust/issues/detail?id=5
  const static int N = 100;
  using T            = LargeStruct<N>;

  const thrust::device_vector<T> dv1;
  const thrust::host_vector<T> hv1;

  REQUIRE((dv1 == hv1));

  const thrust::device_vector<T> dv2(20);
  const thrust::host_vector<T> hv2(20);

  REQUIRE((dv2 == hv2));

  // initialize tofirst element to something nonzero
  T ls;

  for (int i = 0; i < N; i++)
  {
    ls.data[i] = i;
  }

  thrust::device_vector<T> dv3(20, ls);
  thrust::host_vector<T> hv3(20, ls);

  REQUIRE((dv3 == hv3));

  // change first element
  ls.data[0] = -13;

  dv3[2] = ls;
  hv3[2] = ls;

  REQUIRE((dv3 == hv3));
}

template <typename Vector>
void test_vector_reversed()
{
  Vector v{0, 1, 2};

  REQUIRE(3 == v.rend() - v.rbegin());
  REQUIRE(3 == static_cast<const Vector&>(v).rend() - static_cast<const Vector&>(v).rbegin());
  REQUIRE(3 == v.crend() - v.crbegin());

  REQUIRE(2 == *v.rbegin());
  REQUIRE(2 == *static_cast<const Vector&>(v).rbegin());
  REQUIRE(2 == *v.crbegin());

  REQUIRE(1 == *(v.rbegin() + 1));
  REQUIRE(0 == *(v.rbegin() + 2));

  REQUIRE(0 == *(v.rend() - 1));
  REQUIRE(1 == *(v.rend() - 2));
}
DECLARE_VECTOR_UNITTEST(test_vector_reversed);

template <class Vector>
void test_vector_move()
{
  // test move construction
  Vector v1{0, 1, 2};

  const auto ptr1  = v1.data();
  const auto size1 = v1.size();

  Vector v2(std::move(v1));
  const auto ptr2  = v2.data();
  const auto size2 = v2.size();

  // ensure v1 was left empty
  REQUIRE(v1.empty()); // NOLINT(bugprone-use-after-move)

  // ensure v2 received the data from before
  Vector ref{0, 1, 2};
  REQUIRE(v2 == ref);
  REQUIRE(size1 == size2);

  // ensure v2 received the pointer from before
  REQUIRE(ptr1 == ptr2);

  // test move assignment
  Vector v3{3, 4, 5};

  const auto ptr3  = v3.data();
  const auto size3 = v3.size();

  v2               = std::move(v3);
  const auto ptr4  = v2.data();
  const auto size4 = v2.size();

  // ensure v3 was left empty
  REQUIRE(v3.empty()); // NOLINT(bugprone-use-after-move)

  // ensure v2 received the data from before
  ref = {3, 4, 5};
  REQUIRE(v2 == ref);
  REQUIRE(size3 == size4);

  // ensure v2 received the pointer from before
  REQUIRE(ptr3 == ptr4);
}
DECLARE_VECTOR_UNITTEST(test_vector_move);

struct IntWithInit
{
  int value = 42;
};

TEST_CASE("TestVectorDefaultInitCtor", "[vector]")
{
  // trivially-constructible type: just compilation test, since we cannot check that initialization was skipped
  {
    const thrust::host_vector<int> hv(10, thrust::default_init);
    const thrust::device_vector<int> dv(10, thrust::default_init);
  }

  // non-trivially-constructible type: check that initialization was performed
  {
    const thrust::host_vector<IntWithInit> hv(10, thrust::default_init);
    for (auto e : hv)
    {
      REQUIRE(e.value == 42);
    }

    const thrust::device_vector<IntWithInit> dv(10, thrust::default_init);
    for (auto e : dv)
    {
      REQUIRE(static_cast<IntWithInit>(e).value == 42);
    }
  }
}

TEST_CASE("TestVectorNoInitCtor", "[vector]")
{
  // trivially-constructible type: just compilation test, since we cannot check that initialization was skipped
  {
    const thrust::host_vector<int> hv(10, thrust::no_init);
    const thrust::device_vector<int> dv(10, thrust::no_init);
  }

  // non-trivially-constructible type: those should fail to compile
  // thrust::host_vector<IntWithInit> hv(10, thrust::no_init);
  // thrust::device_vector<IntWithInit> dv(10, thrust::no_init);
}

TEST_CASE("TestVectorDefaultInitResize", "[vector]")
{
  // trivially-constructible type: just compilation test, since we cannot check that initialization was skipped
  {
    thrust::host_vector<int> hv(5);
    hv.resize(10, thrust::default_init);
  }
  {
    thrust::device_vector<int> dv(5);
    dv.resize(10, thrust::default_init);
  }

  // non-trivially-constructible type: check that initialization was performed
  {
    thrust::host_vector<IntWithInit> hv(5);
    hv.resize(10, thrust::default_init);
    for (auto e : hv)
    {
      REQUIRE(e.value == 42);
    }
  }
  {
    thrust::device_vector<IntWithInit> dv(5, thrust::default_init);
    dv.resize(10, thrust::default_init);
    for (auto e : dv)
    {
      REQUIRE(static_cast<IntWithInit>(e).value == 42);
    }
  }
}

TEST_CASE("TestVectorNoInitResize", "[vector]")
{
  // trivially-constructible type: just compilation test, since we cannot check that initialization was skipped
  {
    thrust::host_vector<int> hv(5);
    hv.resize(10, thrust::no_init);
  }
  {
    thrust::device_vector<int> dv(5);
    dv.resize(10, thrust::no_init);
  }

  // non-trivially-constructible type: those should fail to compile
  // thrust::host_vector<IntWithInit>(5).resize(10, thrust::no_init);
  // thrust::device_vector<IntWithInit>(5).resize(10, thrust::no_init);
}

struct RemembersCopy
{
  _CCCL_HOST_DEVICE RemembersCopy()
      : n_(0)
  {
    copied_ = false;
  }

  _CCCL_HOST_DEVICE explicit RemembersCopy(int n)
      : n_(n)
  {
    copied_ = false;
  }
  _CCCL_HOST_DEVICE RemembersCopy(const RemembersCopy& other)
  {
    n_      = other.n_;
    copied_ = true;
  }
  _CCCL_HOST_DEVICE RemembersCopy& operator=(const RemembersCopy& other)
  {
    n_      = other.n_;
    copied_ = true;
    return *this;
  }
  _CCCL_HOST_DEVICE bool copied() const
  {
    return copied_;
  }

  int n_;
  bool copied_;
};

// functor used to count if elements are copied
struct is_copied
{
  _CCCL_HOST_DEVICE bool operator()(const RemembersCopy& x) const
  {
    return x.copied();
  }
};

TEST_CASE("TestVectorEmplaceBackDoesNotCopy", "[vector]")
{
  using T = RemembersCopy;

  thrust::host_vector<T> v_h;
  v_h.emplace_back(42);

  thrust::device_vector<T> v_d;
  v_d.emplace_back(42);

  REQUIRE(v_h[0].copied() == false);
  // Verbose but necessary: the flag is on device
  // int n_copied = thrust::count_if(v_d.begin(), v_d.end(), is_copied{}); wrong: this passes copies to the algo !
  const T* first = thrust::raw_pointer_cast(v_d.data());
  // Operate on raw pointer instead
  const int n_copied = thrust::count_if(thrust::device, first, first + v_d.size(), is_copied{});
  REQUIRE(n_copied == 0);
}

enum class Loc
{
  None,
  Host,
  Device
};

struct RemembersConstructionLocation
{
  _CCCL_HOST_DEVICE RemembersConstructionLocation()
      : n_(0)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (loc_ = Loc::Device;), (loc_ = Loc::Host;));
  }

  _CCCL_HOST_DEVICE explicit RemembersConstructionLocation(int n)
      : n_(n)
  {
    NV_IF_TARGET(NV_IS_DEVICE, (loc_ = Loc::Device;), (loc_ = Loc::Host;));
  }

  _CCCL_HOST_DEVICE bool constructed_on_host() const
  {
    return loc_ == Loc::Host;
  }
  _CCCL_HOST_DEVICE bool constructed_on_device() const
  {
    return loc_ == Loc::Device;
  }

  int n_;
  Loc loc_ = Loc::None;
};

// functor used to count if elements are constructed on device
struct IsConstructedOnDevice
{
  _CCCL_HOST_DEVICE bool operator()(const RemembersConstructionLocation& x) const
  {
    return x.constructed_on_device();
  }
};

TEST_CASE("TestVectorEmplaceBackConstructsInTheRightLocation", "[vector]")
{
  using T = RemembersConstructionLocation;

  thrust::host_vector<T> v_h;
  v_h.emplace_back(42);
  REQUIRE(v_h[0].constructed_on_host() == true);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

  thrust::device_vector<T> v_d;
  v_d.emplace_back(42);

  const T* first = thrust::raw_pointer_cast(v_d.data());
  const int n_constructed_on_device =
    thrust::count_if(thrust::device, first, first + v_d.size(), IsConstructedOnDevice{});
  REQUIRE(n_constructed_on_device == 1);

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}

struct ThrowsIfBuiltWithInteger
{
  _CCCL_HOST_DEVICE ThrowsIfBuiltWithInteger()
      : n_(0)
  {}

  _CCCL_HOST_DEVICE explicit ThrowsIfBuiltWithInteger(int n)
      : n_(n)
  {
    NV_IF_TARGET(NV_IS_HOST, (throw std::runtime_error("can't be built like this!");));
  }

  int n_;
};

TEST_CASE("TestVectorEmplaceBackThrowsIfCtorThrows", "[vector]")
{
  using T = ThrowsIfBuiltWithInteger;

  thrust::host_vector<T> v_h;
  v_h.emplace_back();
  v_h.emplace_back();
  REQUIRE_THROWS_AS(v_h.emplace_back(42), std::runtime_error);
}

struct HasExplicitCtor
{
  _CCCL_HOST_DEVICE explicit HasExplicitCtor(int n)
      : n_(n)
  {}

private:
  int n_;
};

TEST_CASE("TestVectorEmplaceBackReturnsReference", "[vector]")
{
  using T = HasExplicitCtor;

  thrust::device_vector<T> v_d;
  thrust::host_vector<T> v_h;
  thrust::universal_vector<T> v_u;
  static_assert(cuda::std::is_same_v<decltype(v_d.emplace_back(42)), thrust::device_vector<T>::reference>);
  static_assert(cuda::std::is_same_v<decltype(v_h.emplace_back(42)), thrust::host_vector<T>::reference>);
  static_assert(cuda::std::is_same_v<decltype(v_u.emplace_back(42)), thrust::universal_vector<T>::reference>);
}

struct HasMultiArgumentCtor
{
  _CCCL_HOST_DEVICE HasMultiArgumentCtor()
      : n0_(0)
      , n1_(0)
  {}

  _CCCL_HOST_DEVICE HasMultiArgumentCtor(int n0, int n1)
      : n0_(n0)
      , n1_(n1)
  {}

  _CCCL_HOST_DEVICE int sum() const
  {
    return n0_ + n1_;
  }

private:
  int n0_;
  int n1_;
};

struct Sum
{
  _CCCL_HOST_DEVICE int operator()(const HasMultiArgumentCtor& x)
  {
    return x.sum();
  }
};

TEST_CASE("TestVectorEmplaceWorksWithMultiArgumentCtor", "[vector]")
{
  using T = HasMultiArgumentCtor;

  thrust::device_vector<T> v_d;
  thrust::host_vector<T> v_h;
  thrust::universal_vector<T> v_u;

  v_d.emplace_back(41, 42);
  v_h.emplace_back(41, 42);
  v_u.emplace_back(41, 42);

  int sum_d = thrust::transform_reduce(thrust::device, v_d.begin(), v_d.end(), Sum{}, 0, ::cuda::std::plus<int>());

  REQUIRE(sum_d == 41 + 42);
  REQUIRE(v_h[0].sum() == 41 + 42);
  REQUIRE(v_u[0].sum() == 41 + 42);
}

template <typename T>
struct small_allocator : std::allocator<T>
{
  std::size_t max_size() const
  {
    return 8;
  }
};

using small_vector = thrust::host_vector<int, small_allocator<int>>;

TEST_CASE("TestVectorInsertionAtMaxSize", "[vector]")
{
  small_vector v(8); // size() == capacity() == max_size()

  REQUIRE_THROWS_AS(v.push_back(1), std::length_error);
  REQUIRE_THROWS_AS(v.emplace_back(1), std::length_error);
  REQUIRE_THROWS_AS(v.resize(9), std::length_error);
  REQUIRE(v.size() == 8);
}

TEST_CASE("TestVectorBulkInsertionBeyondMaxSize", "[vector]")
{
  small_vector v(3);
  const thrust::host_vector<int> src(6, 1); // 3 + 6 > max_size()

  REQUIRE_THROWS_AS(v.insert(v.end(), 6, 1), std::length_error);
  REQUIRE_THROWS_AS(v.insert(v.end(), src.begin(), src.end()), std::length_error);
  REQUIRE_THROWS_AS(v.resize(9, 1), std::length_error);
  REQUIRE(v.size() == 3);
}

TEST_CASE("TestVectorGrowthSaturatesAtMaxSize", "[vector]")
{
  // doubling capacity 5 would give 10 > max_size(), so growth must stop at 8
  const std::vector<int> one(1, 1);

  small_vector a(5);
  a.push_back(1);
  REQUIRE(a.capacity() == 8);

  small_vector b(5);
  b.emplace_back(1);
  REQUIRE(b.capacity() == 8);

  small_vector c(5);
  c.insert(c.end(), one.begin(), one.end());
  REQUIRE(c.capacity() == 8);

  small_vector d(5);
  d.resize(6);
  REQUIRE(d.capacity() == 8);
}

struct RemembersConstructionType
{
  _CCCL_HOST_DEVICE RemembersConstructionType()
      : state_(0) {};
  _CCCL_HOST_DEVICE RemembersConstructionType(RemembersConstructionType& other)
      : copy_cted_(true)
      , state_(other.state_) {};
  _CCCL_HOST_DEVICE RemembersConstructionType(const RemembersConstructionType& other)
      : const_copy_cted_(true)
      , state_(other.state_) {};
  _CCCL_HOST_DEVICE RemembersConstructionType(RemembersConstructionType&& other)
      : move_cted_(true)
      , state_(other.state_) {};
  _CCCL_HOST_DEVICE RemembersConstructionType(const RemembersConstructionType&& other)
      : const_move_cted_(true)
      , state_(other.state_) {};

  bool move_cted() const
  {
    return move_cted_;
  }
  bool copy_cted() const
  {
    return copy_cted_;
  }
  bool const_move_cted() const
  {
    return const_move_cted_;
  }
  bool const_copy_cted() const
  {
    return const_copy_cted_;
  }

  bool copy_cted_       = false;
  bool move_cted_       = false;
  bool const_copy_cted_ = false;
  bool const_move_cted_ = false;
  int state_;
};

// Helper to read a flag in place on the device: copying the element to host would run a ctor and reset it
using construction_flag = bool RemembersConstructionType::*;
struct flag_is_set
{
  construction_flag flag;
  _CCCL_HOST_DEVICE bool operator()(const RemembersConstructionType& t) const
  {
    return t.*flag;
  }
};
bool device_flag(const thrust::device_vector<RemembersConstructionType>& v, size_t i, construction_flag flag)
{
  const auto* p = thrust::raw_pointer_cast(v.data()) + i;
  return thrust::count_if(thrust::device, p, p + 1, flag_is_set{flag}) == 1;
}

TEST_CASE("TestEmplaceBackCallsRightConstructor", "[vector]")
{
  using T = RemembersConstructionType;

  {
    thrust::host_vector<T> v_h;

    T x;
    const T cx;

    // Interleave tests and emplaces to avoid reallocation side effects.
    v_h.emplace_back(x);
    REQUIRE(v_h[0].copy_cted() == true);
    v_h.emplace_back(cx);
    REQUIRE(v_h[1].const_copy_cted() == true);
    v_h.emplace_back(cuda::std::move(x));
    REQUIRE(v_h[2].move_cted() == true);
    v_h.emplace_back(cuda::std::move(cx));
    REQUIRE(v_h[3].const_move_cted() == true);
  }

  {
    thrust::device_vector<T> v_d;

    T x;
    const T cx;

    v_d.emplace_back(x);
    REQUIRE(device_flag(v_d, 0, &T::copy_cted_));
    v_d.emplace_back(cx);
    REQUIRE(device_flag(v_d, 1, &T::const_copy_cted_));
    v_d.emplace_back(cuda::std::move(x));
    REQUIRE(device_flag(v_d, 2, &T::move_cted_));
    v_d.emplace_back(cuda::std::move(cx));
    REQUIRE(device_flag(v_d, 3, &T::const_move_cted_));
  }
}
