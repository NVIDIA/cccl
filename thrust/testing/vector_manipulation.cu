#include <thrust/device_malloc_allocator.h>

#include <vector>

#include <unittest/unittest.h>

template <class Vector>
void TestVectorManipulation(size_t n)
{
  using Iterator = typename Vector::iterator;
  using T        = typename Vector::value_type;

  thrust::host_vector<T> src = unittest::random_samples<T>(n);
  REQUIRE(src.size() == n);

  // basic initialization
  Vector test0(n);
  Vector test1(n, T(3));
  REQUIRE(test0.size() == n);
  REQUIRE(test1.size() == n);
  REQUIRE(test1 == std::vector<T>(n, T(3)));

  // initializing from other vector
  std::vector<T> stl_vector(src.begin(), src.end());
  Vector cpy0 = src;
  Vector cpy1(stl_vector);
  Vector cpy2(stl_vector.begin(), stl_vector.end());
  REQUIRE(cpy0 == src);
  REQUIRE(cpy1 == src);
  REQUIRE(cpy2 == src);

  // resizing
  Vector vec1(src);
  vec1.resize(n + 3);
  REQUIRE(vec1.size() == n + 3);
  vec1.resize(n);
  REQUIRE(vec1.size() == n);
  REQUIRE(vec1 == src);

  vec1.resize(n + 20, T(11));
  Vector tail(vec1.begin() + n, vec1.end());
  REQUIRE(tail == std::vector<T>(20, T(11)));

  // shrinking a vector should not invalidate iterators
  const Iterator first = vec1.begin();
  vec1.resize(10);
  ASSERT_EQUAL_QUIET(first, vec1.begin());

  vec1.resize(0);
  REQUIRE(vec1.size() == 0lu);
  REQUIRE(vec1.empty());
  vec1.resize(10);
  REQUIRE(vec1.size() == 10lu);
  vec1.clear();
  REQUIRE(vec1.size() == 0lu);
  vec1.resize(5);
  REQUIRE(vec1.size() == 5lu);

  // push_back
  Vector vec2;
  for (size_t i = 0; i < 10; ++i)
  {
    REQUIRE(vec2.size() == i);
    vec2.push_back(T(i));
    REQUIRE(vec2.size() == i + 1);
    for (size_t j = 0; j <= i; j++)
    {
      REQUIRE(vec2[j] == T(j));
    }
    REQUIRE(vec2.back() == T(i));
  }

  // pop_back
  for (size_t i = 10; i > 0; --i)
  {
    REQUIRE(vec2.size() == i);
    REQUIRE(vec2.back() == T(i - 1));
    vec2.pop_back();
    REQUIRE(vec2.size() == i - 1);
    for (size_t j = 0; j < i; j++)
    {
      REQUIRE(vec2[j] == T(j));
    }
  }

  // TODO test swap, erase(pos), erase(begin, end)
}

template <typename T>
void TestVectorManipulationHost(size_t n)
{
  TestVectorManipulation<thrust::host_vector<T>>(n);
}
DECLARE_VARIABLE_UNITTEST(TestVectorManipulationHost);

template <typename T>
void TestVectorManipulationDevice(size_t n)
{
  TestVectorManipulation<thrust::device_vector<T>>(n);
}
DECLARE_VARIABLE_UNITTEST(TestVectorManipulationDevice);
