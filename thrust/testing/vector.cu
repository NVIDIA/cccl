#include <thrust/detail/config.h>

#include <thrust/device_malloc_allocator.h>
#include <thrust/sequence.h>

#include <initializer_list>
#include <limits>
#include <list>
#include <utility>
#include <vector>

#include <unittest/unittest.h>

template <class Vector>
void TestVectorZeroSize()
{
  Vector v;
  ASSERT_EQUAL(v.size(), 0lu);
  ASSERT_EQUAL((v.begin() == v.end()), true);
}
DECLARE_VECTOR_UNITTEST(TestVectorZeroSize);

void TestVectorBool()
{
  thrust::host_vector<bool> h{true, false, true};
  thrust::device_vector<bool> d{true, false, true};

  thrust::host_vector<bool> h_ref{true, false, true};
  thrust::device_vector<bool> d_ref{true, false, true};
  ASSERT_EQUAL(h, h_ref);
  ASSERT_EQUAL(d, d_ref);
}
DECLARE_UNITTEST(TestVectorBool);

template <class Vector>
void TestVectorInitializerList()
{
  Vector v{1, 2, 3};
  ASSERT_EQUAL(v.size(), 3lu);
  Vector ref{1, 2, 3};
  ASSERT_EQUAL(v, ref);

  v = {1, 2, 3, 4};
  ASSERT_EQUAL(v.size(), 4lu);
  Vector v_ref = {1, 2, 3, 4};
  ASSERT_EQUAL(v, v_ref);

  const auto alloc = v.get_allocator();
  Vector v2{{1, 2, 3}, alloc};
  ASSERT_EQUAL(v2.size(), 3lu);
  Vector v2_ref = {1, 2, 3};
  ASSERT_EQUAL(v2, v2_ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorInitializerList);

template <class Vector>
void TestVectorFrontBack()
{
  using T = typename Vector::value_type;

  Vector v{0, 1, 2};

  ASSERT_EQUAL(v.front(), T(0));
  ASSERT_EQUAL(v.back(), T(2));
}
DECLARE_VECTOR_UNITTEST(TestVectorFrontBack);

template <class Vector>
void TestVectorData()
{
  using PointerT      = typename Vector::pointer;
  using PointerConstT = typename Vector::const_pointer;

  Vector v{0, 1, 2};

  ASSERT_EQUAL(0, *v.data());
  ASSERT_EQUAL(1, *(v.data() + 1));
  ASSERT_EQUAL(2, *(v.data() + 2));
  ASSERT_EQUAL(PointerT(&v.front()), v.data());
  ASSERT_EQUAL(PointerT(&*v.begin()), v.data());
  ASSERT_EQUAL(PointerT(&v[0]), v.data());

  const Vector& c_v = v;

  ASSERT_EQUAL(0, *c_v.data());
  ASSERT_EQUAL(1, *(c_v.data() + 1));
  ASSERT_EQUAL(2, *(c_v.data() + 2));
  ASSERT_EQUAL(PointerConstT(&c_v.front()), c_v.data());
  ASSERT_EQUAL(PointerConstT(&*c_v.begin()), c_v.data());
  ASSERT_EQUAL(PointerConstT(&c_v[0]), c_v.data());
}
DECLARE_VECTOR_UNITTEST(TestVectorData);

template <class Vector>
void TestVectorElementAssignment()
{
  Vector v{0, 1, 2};

  Vector ref{0, 1, 2};
  ASSERT_EQUAL(v, ref);

  v   = {10, 11, 12};
  ref = {10, 11, 12};
  ASSERT_EQUAL(v, ref);

  Vector w = v;
  ASSERT_EQUAL(v, w);
}
DECLARE_VECTOR_UNITTEST(TestVectorElementAssignment);

template <class Vector>
void TestVectorFromSTLVector()
{
  using T = typename Vector::value_type;

  std::vector<T> stl_vector{0, 1, 2};

  thrust::host_vector<T> v(stl_vector);

  ASSERT_EQUAL(v.size(), 3lu);
  thrust::host_vector<T> ref{0, 1, 2};
  ASSERT_EQUAL(v, ref);

  v = stl_vector;

  ASSERT_EQUAL(v.size(), 3lu);
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorFromSTLVector);

template <class Vector>
void TestVectorFillAssign()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> v;
  v.assign(3, 13);

  ASSERT_EQUAL(v.size(), 3lu);
  thrust::host_vector<T> ref{13, 13, 13};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorFillAssign);

template <class Vector>
void TestVectorAssignFromSTLVector()
{
  using T = typename Vector::value_type;

  std::vector<T> stl_vector{0, 1, 2};

  thrust::host_vector<T> v;
  v.assign(stl_vector.begin(), stl_vector.end());

  ASSERT_EQUAL(v.size(), 3lu);
  thrust::host_vector<T> ref{0, 1, 2};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromSTLVector);

template <class Vector>
void TestVectorFromBiDirectionalIterator()
{
  using T = typename Vector::value_type;

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v(stl_list.begin(), stl_list.end());

  ASSERT_EQUAL(v.size(), 3lu);
  Vector ref{0, 1, 2};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorFromBiDirectionalIterator);

template <class Vector>
void TestVectorAssignFromBiDirectionalIterator()
{
  using T = typename Vector::value_type;

  std::list<T> stl_list;
  stl_list.push_back(0);
  stl_list.push_back(1);
  stl_list.push_back(2);

  Vector v;
  v.assign(stl_list.begin(), stl_list.end());

  ASSERT_EQUAL(v.size(), 3lu);
  Vector ref{0, 1, 2};
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromBiDirectionalIterator);

template <class Vector>
void TestVectorAssignFromHostVector()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> h{0, 1, 2};

  Vector v;
  v.assign(h.begin(), h.end());

  ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromHostVector);

template <class Vector>
void TestVectorToAndFromHostVector()
{
  using T = typename Vector::value_type;

  thrust::host_vector<T> h{0, 1, 2};

  Vector v(h);

  ASSERT_EQUAL(v, h);

  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_CLANG("-Wself-assign")
  v = v;
  _CCCL_DIAG_POP

  ASSERT_EQUAL(v, h);

  v = {10, 11, 12};
  Vector v_ref{10, 11, 12};
  ASSERT_EQUAL(v, v_ref);

  Vector h_ref{0, 1, 2};
  ASSERT_EQUAL(h, h_ref);

  h = v;

  ASSERT_EQUAL(v, h);

  h[1] = 11;

  v = h;

  ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorToAndFromHostVector);

template <class Vector>
void TestVectorAssignFromDeviceVector()
{
  using T = typename Vector::value_type;

  thrust::device_vector<T> d{0, 1, 2};

  Vector v;
  v.assign(d.begin(), d.end());

  ASSERT_EQUAL(v, d);
}
DECLARE_VECTOR_UNITTEST(TestVectorAssignFromDeviceVector);

template <class Vector>
void TestVectorToAndFromDeviceVector()
{
  using T = typename Vector::value_type;

  thrust::device_vector<T> h{0, 1, 2};

  Vector v(h);

  ASSERT_EQUAL(v, h);

  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_CLANG("-Wself-assign")
  v = v;
  _CCCL_DIAG_POP

  ASSERT_EQUAL(v, h);

  v = {10, 11, 12};
  Vector v_ref{10, 11, 12};
  ASSERT_EQUAL(v, v_ref);

  Vector h_ref{0, 1, 2};
  ASSERT_EQUAL(h, h_ref);

  h = v;

  ASSERT_EQUAL(v, h);

  h[1] = 11;

  v = h;

  ASSERT_EQUAL(v, h);
}
DECLARE_VECTOR_UNITTEST(TestVectorToAndFromDeviceVector);

template <class Vector>
void TestVectorWithInitialValue()
{
  using T = typename Vector::value_type;

  const T init = 17;

  Vector v(3, init);

  ASSERT_EQUAL(v.size(), 3lu);
  Vector ref(3, init);
  ASSERT_EQUAL(v, ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorWithInitialValue);

template <class Vector>
void TestVectorSwap()
{
  Vector v{0, 1, 2};
  Vector u{10, 11, 12};

  v.swap(u);

  Vector u_ref{0, 1, 2};
  ASSERT_EQUAL(u, u_ref);

  Vector v_ref{10, 11, 12};
  ASSERT_EQUAL(v, v_ref);
}
DECLARE_VECTOR_UNITTEST(TestVectorSwap);

template <class Vector>
void TestVectorErasePosition()
{
  Vector v{0, 1, 2, 3, 4};

  v.erase(v.begin() + 2);

  ASSERT_EQUAL(v.size(), 4lu);
  Vector ref{0, 1, 3, 4};
  ASSERT_EQUAL(v, ref);

  v.erase(v.begin() + 0);

  ASSERT_EQUAL(v.size(), 3lu);
  ref = {1, 3, 4};
  ASSERT_EQUAL(v, ref);

  v.erase(v.begin() + 2);

  ASSERT_EQUAL(v.size(), 2lu);
  ref = {1, 3};
  ASSERT_EQUAL(v, ref);

  v.erase(v.begin() + 1);

  ASSERT_EQUAL(v.size(), 1lu);
  ASSERT_EQUAL(v[0], 1);

  v.erase(v.begin() + 0);

  ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorErasePosition);

template <class Vector>
void TestVectorEraseRange()
{
  Vector v{0, 1, 2, 3, 4, 5};

  v.erase(v.begin() + 1, v.begin() + 3);

  ASSERT_EQUAL(v.size(), 4lu);
  Vector ref{0, 3, 4, 5};
  ASSERT_EQUAL(v, ref);

  v.erase(v.begin() + 2, v.end());

  ASSERT_EQUAL(v.size(), 2lu);
  ref = {0, 3};
  ASSERT_EQUAL(v, ref);

  v.erase(v.begin() + 0, v.begin() + 1);

  ASSERT_EQUAL(v.size(), 1lu);
  ASSERT_EQUAL(v[0], 3);

  v.erase(v.begin(), v.end());

  ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorEraseRange);

void TestVectorEquality()
{
  thrust::host_vector<int> h_a{0, 1, 2};
  thrust::host_vector<int> h_b{0, 1, 3};
  thrust::host_vector<int> h_c(3);

  thrust::device_vector<int> d_a{0, 1, 2};
  thrust::device_vector<int> d_b{0, 1, 3};
  thrust::device_vector<int> d_c(3);

  std::vector<int> s_a{0, 1, 2};
  std::vector<int> s_b{0, 1, 3};
  std::vector<int> s_c(3);

  ASSERT_EQUAL((h_a == h_a), true);
  ASSERT_EQUAL((h_a == d_a), true);
  ASSERT_EQUAL((d_a == h_a), true);
  ASSERT_EQUAL((d_a == d_a), true);
  ASSERT_EQUAL((h_b == h_b), true);
  ASSERT_EQUAL((h_b == d_b), true);
  ASSERT_EQUAL((d_b == h_b), true);
  ASSERT_EQUAL((d_b == d_b), true);
  ASSERT_EQUAL((h_c == h_c), true);
  ASSERT_EQUAL((h_c == d_c), true);
  ASSERT_EQUAL((d_c == h_c), true);
  ASSERT_EQUAL((d_c == d_c), true);

  // test vector vs device_vector
  ASSERT_EQUAL((s_a == d_a), true);
  ASSERT_EQUAL((d_a == s_a), true);
  ASSERT_EQUAL((s_b == d_b), true);
  ASSERT_EQUAL((d_b == s_b), true);
  ASSERT_EQUAL((s_c == d_c), true);
  ASSERT_EQUAL((d_c == s_c), true);

  // test vector vs host_vector
  ASSERT_EQUAL((s_a == h_a), true);
  ASSERT_EQUAL((h_a == s_a), true);
  ASSERT_EQUAL((s_b == h_b), true);
  ASSERT_EQUAL((h_b == s_b), true);
  ASSERT_EQUAL((s_c == h_c), true);
  ASSERT_EQUAL((h_c == s_c), true);

  ASSERT_EQUAL((h_a == h_b), false);
  ASSERT_EQUAL((h_a == d_b), false);
  ASSERT_EQUAL((d_a == h_b), false);
  ASSERT_EQUAL((d_a == d_b), false);
  ASSERT_EQUAL((h_b == h_a), false);
  ASSERT_EQUAL((h_b == d_a), false);
  ASSERT_EQUAL((d_b == h_a), false);
  ASSERT_EQUAL((d_b == d_a), false);
  ASSERT_EQUAL((h_a == h_c), false);
  ASSERT_EQUAL((h_a == d_c), false);
  ASSERT_EQUAL((d_a == h_c), false);
  ASSERT_EQUAL((d_a == d_c), false);
  ASSERT_EQUAL((h_c == h_a), false);
  ASSERT_EQUAL((h_c == d_a), false);
  ASSERT_EQUAL((d_c == h_a), false);
  ASSERT_EQUAL((d_c == d_a), false);
  ASSERT_EQUAL((h_b == h_c), false);
  ASSERT_EQUAL((h_b == d_c), false);
  ASSERT_EQUAL((d_b == h_c), false);
  ASSERT_EQUAL((d_b == d_c), false);
  ASSERT_EQUAL((h_c == h_b), false);
  ASSERT_EQUAL((h_c == d_b), false);
  ASSERT_EQUAL((d_c == h_b), false);
  ASSERT_EQUAL((d_c == d_b), false);

  // test vector vs device_vector
  ASSERT_EQUAL((s_a == d_b), false);
  ASSERT_EQUAL((d_a == s_b), false);
  ASSERT_EQUAL((s_b == d_a), false);
  ASSERT_EQUAL((d_b == s_a), false);
  ASSERT_EQUAL((s_a == d_c), false);
  ASSERT_EQUAL((d_a == s_c), false);
  ASSERT_EQUAL((s_c == d_a), false);
  ASSERT_EQUAL((d_c == s_a), false);
  ASSERT_EQUAL((s_b == d_c), false);
  ASSERT_EQUAL((d_b == s_c), false);
  ASSERT_EQUAL((s_c == d_b), false);
  ASSERT_EQUAL((d_c == s_b), false);

  // test vector vs host_vector
  ASSERT_EQUAL((s_a == h_b), false);
  ASSERT_EQUAL((h_a == s_b), false);
  ASSERT_EQUAL((s_b == h_a), false);
  ASSERT_EQUAL((h_b == s_a), false);
  ASSERT_EQUAL((s_a == h_c), false);
  ASSERT_EQUAL((h_a == s_c), false);
  ASSERT_EQUAL((s_c == h_a), false);
  ASSERT_EQUAL((h_c == s_a), false);
  ASSERT_EQUAL((s_b == h_c), false);
  ASSERT_EQUAL((h_b == s_c), false);
  ASSERT_EQUAL((s_c == h_b), false);
  ASSERT_EQUAL((h_c == s_b), false);
}
DECLARE_UNITTEST(TestVectorEquality);

void TestVectorInequality()
{
  thrust::host_vector<int> h_a{0, 1, 2};
  thrust::host_vector<int> h_b{0, 1, 3};
  thrust::host_vector<int> h_c(3);

  thrust::device_vector<int> d_a{0, 1, 2};
  thrust::device_vector<int> d_b{0, 1, 3};
  thrust::device_vector<int> d_c(3);

  std::vector<int> s_a{0, 1, 2};
  std::vector<int> s_b{0, 1, 3};
  std::vector<int> s_c(3);

  ASSERT_EQUAL((h_a != h_a), false);
  ASSERT_EQUAL((h_a != d_a), false);
  ASSERT_EQUAL((d_a != h_a), false);
  ASSERT_EQUAL((d_a != d_a), false);
  ASSERT_EQUAL((h_b != h_b), false);
  ASSERT_EQUAL((h_b != d_b), false);
  ASSERT_EQUAL((d_b != h_b), false);
  ASSERT_EQUAL((d_b != d_b), false);
  ASSERT_EQUAL((h_c != h_c), false);
  ASSERT_EQUAL((h_c != d_c), false);
  ASSERT_EQUAL((d_c != h_c), false);
  ASSERT_EQUAL((d_c != d_c), false);

  // test vector vs device_vector
  ASSERT_EQUAL((s_a != d_a), false);
  ASSERT_EQUAL((d_a != s_a), false);
  ASSERT_EQUAL((s_b != d_b), false);
  ASSERT_EQUAL((d_b != s_b), false);
  ASSERT_EQUAL((s_c != d_c), false);
  ASSERT_EQUAL((d_c != s_c), false);

  // test vector vs host_vector
  ASSERT_EQUAL((s_a != h_a), false);
  ASSERT_EQUAL((h_a != s_a), false);
  ASSERT_EQUAL((s_b != h_b), false);
  ASSERT_EQUAL((h_b != s_b), false);
  ASSERT_EQUAL((s_c != h_c), false);
  ASSERT_EQUAL((h_c != s_c), false);

  ASSERT_EQUAL((h_a != h_b), true);
  ASSERT_EQUAL((h_a != d_b), true);
  ASSERT_EQUAL((d_a != h_b), true);
  ASSERT_EQUAL((d_a != d_b), true);
  ASSERT_EQUAL((h_b != h_a), true);
  ASSERT_EQUAL((h_b != d_a), true);
  ASSERT_EQUAL((d_b != h_a), true);
  ASSERT_EQUAL((d_b != d_a), true);
  ASSERT_EQUAL((h_a != h_c), true);
  ASSERT_EQUAL((h_a != d_c), true);
  ASSERT_EQUAL((d_a != h_c), true);
  ASSERT_EQUAL((d_a != d_c), true);
  ASSERT_EQUAL((h_c != h_a), true);
  ASSERT_EQUAL((h_c != d_a), true);
  ASSERT_EQUAL((d_c != h_a), true);
  ASSERT_EQUAL((d_c != d_a), true);
  ASSERT_EQUAL((h_b != h_c), true);
  ASSERT_EQUAL((h_b != d_c), true);
  ASSERT_EQUAL((d_b != h_c), true);
  ASSERT_EQUAL((d_b != d_c), true);
  ASSERT_EQUAL((h_c != h_b), true);
  ASSERT_EQUAL((h_c != d_b), true);
  ASSERT_EQUAL((d_c != h_b), true);
  ASSERT_EQUAL((d_c != d_b), true);

  // test vector vs device_vector
  ASSERT_EQUAL((s_a != d_b), true);
  ASSERT_EQUAL((d_a != s_b), true);
  ASSERT_EQUAL((s_b != d_a), true);
  ASSERT_EQUAL((d_b != s_a), true);
  ASSERT_EQUAL((s_a != d_c), true);
  ASSERT_EQUAL((d_a != s_c), true);
  ASSERT_EQUAL((s_c != d_a), true);
  ASSERT_EQUAL((d_c != s_a), true);
  ASSERT_EQUAL((s_b != d_c), true);
  ASSERT_EQUAL((d_b != s_c), true);
  ASSERT_EQUAL((s_c != d_b), true);
  ASSERT_EQUAL((d_c != s_b), true);

  // test vector vs host_vector
  ASSERT_EQUAL((s_a != h_b), true);
  ASSERT_EQUAL((h_a != s_b), true);
  ASSERT_EQUAL((s_b != h_a), true);
  ASSERT_EQUAL((h_b != s_a), true);
  ASSERT_EQUAL((s_a != h_c), true);
  ASSERT_EQUAL((h_a != s_c), true);
  ASSERT_EQUAL((s_c != h_a), true);
  ASSERT_EQUAL((h_c != s_a), true);
  ASSERT_EQUAL((s_b != h_c), true);
  ASSERT_EQUAL((h_b != s_c), true);
  ASSERT_EQUAL((s_c != h_b), true);
  ASSERT_EQUAL((h_c != s_b), true);
}
DECLARE_UNITTEST(TestVectorInequality);

template <class Vector>
void TestVectorResizing()
{
  Vector v;

  v.resize(3);

  ASSERT_EQUAL(v.size(), 3lu);

  v = {0, 1, 2};
  v.resize(5);

  ASSERT_EQUAL(v.size(), 5lu);

  Vector ref{0, 1, 2, v[3], v[4]};
  ASSERT_EQUAL(v, ref);

  v[3] = 3;
  v[4] = 4;

  v.resize(4);

  ASSERT_EQUAL(v.size(), 4lu);

  ref = {0, 1, 2, 3};
  ASSERT_EQUAL(v, ref);

  v.resize(0);

  ASSERT_EQUAL(v.size(), 0lu);

// TODO remove this WAR
#if defined(__CUDACC__) && CUDART_VERSION == 3000
  // depending on sizeof(T), we will receive one
  // of two possible exceptions
  try
  {
    v.resize(std::numeric_limits<size_t>::max());
  }
  catch (std::length_error e)
  {}
  catch (std::bad_alloc e)
  {
    // reset the CUDA error
    cudaGetLastError();
  } // end catch
#endif // defined(__CUDACC__) && CUDART_VERSION==3000

  ASSERT_EQUAL(v.size(), 0lu);
}
DECLARE_VECTOR_UNITTEST(TestVectorResizing);

template <class Vector>
void TestVectorReserving()
{
  Vector v;

  v.reserve(3);

  ASSERT_GEQUAL(v.capacity(), 3lu);

  size_t old_capacity = v.capacity();

  v.reserve(0);

  ASSERT_EQUAL(v.capacity(), old_capacity);

// TODO remove this WAR
#if defined(__CUDACC__) && CUDART_VERSION == 3000
  try
  {
    v.reserve(std::numeric_limits<size_t>::max());
  }
  catch (std::length_error e)
  {}
  catch (std::bad_alloc e)
  {}
#endif // defined(__CUDACC__) && CUDART_VERSION==3000

  ASSERT_EQUAL(v.capacity(), old_capacity);
}
DECLARE_VECTOR_UNITTEST(TestVectorReserving)

template <class Vector>
void TestVectorUninitialisedCopy()
{
  thrust::device_vector<int> v;
  std::vector<int> std_vector;

  v = std_vector;

  ASSERT_EQUAL(v.size(), static_cast<size_t>(0));
}
DECLARE_VECTOR_UNITTEST(TestVectorUninitialisedCopy);

template <class Vector>
void TestVectorShrinkToFit()
{
  using T = typename Vector::value_type;

  Vector v;

  v.reserve(200);

  ASSERT_GEQUAL(v.capacity(), 200lu);

  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  v.shrink_to_fit();

  ASSERT_EQUAL(T(1), v[0]);
  ASSERT_EQUAL(T(2), v[1]);
  ASSERT_EQUAL(T(3), v[2]);
  ASSERT_EQUAL(3lu, v.size());
  ASSERT_EQUAL(3lu, v.capacity());
}
DECLARE_VECTOR_UNITTEST(TestVectorShrinkToFit)

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

void TestVectorContainingLargeType()
{
  // Thrust issue #5
  // http://code.google.com/p/thrust/issues/detail?id=5
  const static int N = 100;
  using T            = LargeStruct<N>;

  thrust::device_vector<T> dv1;
  thrust::host_vector<T> hv1;

  ASSERT_EQUAL_QUIET(dv1, hv1);

  thrust::device_vector<T> dv2(20);
  thrust::host_vector<T> hv2(20);

  ASSERT_EQUAL_QUIET(dv2, hv2);

  // initialize tofirst element to something nonzero
  T ls;

  for (int i = 0; i < N; i++)
  {
    ls.data[i] = i;
  }

  thrust::device_vector<T> dv3(20, ls);
  thrust::host_vector<T> hv3(20, ls);

  ASSERT_EQUAL_QUIET(dv3, hv3);

  // change first element
  ls.data[0] = -13;

  dv3[2] = ls;
  hv3[2] = ls;

  ASSERT_EQUAL_QUIET(dv3, hv3);
}
DECLARE_UNITTEST(TestVectorContainingLargeType);

template <typename Vector>
void TestVectorReversed()
{
  Vector v{0, 1, 2};

  ASSERT_EQUAL(3, v.rend() - v.rbegin());
  ASSERT_EQUAL(3, static_cast<const Vector&>(v).rend() - static_cast<const Vector&>(v).rbegin());
  ASSERT_EQUAL(3, v.crend() - v.crbegin());

  ASSERT_EQUAL(2, *v.rbegin());
  ASSERT_EQUAL(2, *static_cast<const Vector&>(v).rbegin());
  ASSERT_EQUAL(2, *v.crbegin());

  ASSERT_EQUAL(1, *(v.rbegin() + 1));
  ASSERT_EQUAL(0, *(v.rbegin() + 2));

  ASSERT_EQUAL(0, *(v.rend() - 1));
  ASSERT_EQUAL(1, *(v.rend() - 2));
}
DECLARE_VECTOR_UNITTEST(TestVectorReversed);

template <class Vector>
void TestVectorMove()
{
  // test move construction
  Vector v1{0, 1, 2};

  const auto ptr1  = v1.data();
  const auto size1 = v1.size();

  Vector v2(std::move(v1));
  const auto ptr2  = v2.data();
  const auto size2 = v2.size();

  // ensure v1 was left empty
  ASSERT_EQUAL(true, v1.empty());

  // ensure v2 received the data from before
  Vector ref{0, 1, 2};
  ASSERT_EQUAL(v2, ref);
  ASSERT_EQUAL(size1, size2);

  // ensure v2 received the pointer from before
  ASSERT_EQUAL(ptr1, ptr2);

  // test move assignment
  Vector v3{3, 4, 5};

  const auto ptr3  = v3.data();
  const auto size3 = v3.size();

  v2               = std::move(v3);
  const auto ptr4  = v2.data();
  const auto size4 = v2.size();

  // ensure v3 was left empty
  ASSERT_EQUAL(true, v3.empty());

  // ensure v2 received the data from before
  ref = {3, 4, 5};
  ASSERT_EQUAL(v2, ref);
  ASSERT_EQUAL(size3, size4);

  // ensure v2 received the pointer from before
  ASSERT_EQUAL(ptr3, ptr4);
}
DECLARE_VECTOR_UNITTEST(TestVectorMove);
