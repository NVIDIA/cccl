#include <thrust/detail/alignment.h>

#include "catch2_test_helper.h"

struct alignof_mock_0
{
  char a;
  char b;
}; // size: 2 * sizeof(char) == alignment: sizeof(char)

struct alignof_mock_1
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int) == alignment: sizeof(int)

struct alignof_mock_2
{
  int n;
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 2 * sizeof(int) == alignment: sizeof(int)

struct alignof_mock_3
{
  char c;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int) == alignment: sizeof(int)

struct alignof_mock_4
{
  char c0;
  // sizeof(int) - sizeof(char) bytes of padding
  int n;
  char c1;
  // sizeof(int) - sizeof(char) bytes of padding
}; // size: 3 * sizeof(int) == alignment: sizeof(int)

struct alignof_mock_5
{
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
  int n;
}; // size: 2 * sizeof(int) == alignment: sizeof(int)

struct alignof_mock_6
{
  int n;
  char c0;
  char c1;
  // sizeof(int) - 2 * sizeof(char) bytes of padding
}; // size: 2 * sizeof(int) == alignment: sizeof(int)

TEST_CASE("alignof_mocks_sizes", "[alignment]")
{
  STATIC_REQUIRE(sizeof(alignof_mock_0) == 2 * sizeof(char));
  STATIC_REQUIRE(sizeof(alignof_mock_1) == 2 * sizeof(int));
  STATIC_REQUIRE(sizeof(alignof_mock_2) == 2 * sizeof(int));
  STATIC_REQUIRE(sizeof(alignof_mock_3) == 2 * sizeof(int));
  STATIC_REQUIRE(sizeof(alignof_mock_4) == 3 * sizeof(int));
  STATIC_REQUIRE(sizeof(alignof_mock_5) == 2 * sizeof(int));
  STATIC_REQUIRE(sizeof(alignof_mock_6) == 2 * sizeof(int));
}

TEST_CASE("alignof", "[alignment]")
{
  STATIC_REQUIRE(alignof(bool) == sizeof(bool));
  STATIC_REQUIRE(alignof(signed char) == sizeof(signed char));
  STATIC_REQUIRE(alignof(unsigned char) == sizeof(unsigned char));
  STATIC_REQUIRE(alignof(char) == sizeof(char));
  STATIC_REQUIRE(alignof(short int) == sizeof(short int));
  STATIC_REQUIRE(alignof(unsigned short int) == sizeof(unsigned short int));
  STATIC_REQUIRE(alignof(int) == sizeof(int));
  STATIC_REQUIRE(alignof(unsigned int) == sizeof(unsigned int));
  STATIC_REQUIRE(alignof(long int) == sizeof(long int));
  STATIC_REQUIRE(alignof(unsigned long int) == sizeof(unsigned long int));
  STATIC_REQUIRE(alignof(long long int) == sizeof(long long int));
  STATIC_REQUIRE(alignof(unsigned long long int) == sizeof(unsigned long long int));
  STATIC_REQUIRE(alignof(float) == sizeof(float));
  STATIC_REQUIRE(alignof(double) == sizeof(double));
  STATIC_REQUIRE(alignof(long double) == sizeof(long double));

  STATIC_REQUIRE(alignof(alignof_mock_0) == sizeof(char));
  STATIC_REQUIRE(alignof(alignof_mock_1) == sizeof(int));
  STATIC_REQUIRE(alignof(alignof_mock_2) == sizeof(int));
  STATIC_REQUIRE(alignof(alignof_mock_3) == sizeof(int));
  STATIC_REQUIRE(alignof(alignof_mock_4) == sizeof(int));
  STATIC_REQUIRE(alignof(alignof_mock_5) == sizeof(int));
  STATIC_REQUIRE(alignof(alignof_mock_6) == sizeof(int));
}

TEST_CASE("alignment_of", "[alignment]")
{
  STATIC_REQUIRE(thrust::detail::alignment_of<bool>::value == sizeof(bool));
  STATIC_REQUIRE(thrust::detail::alignment_of<signed char>::value == sizeof(signed char));
  STATIC_REQUIRE(thrust::detail::alignment_of<unsigned char>::value == sizeof(unsigned char));
  STATIC_REQUIRE(thrust::detail::alignment_of<char>::value == sizeof(char));
  STATIC_REQUIRE(thrust::detail::alignment_of<short int>::value == sizeof(short int));
  STATIC_REQUIRE(thrust::detail::alignment_of<unsigned short int>::value == sizeof(unsigned short int));
  STATIC_REQUIRE(thrust::detail::alignment_of<int>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<unsigned int>::value == sizeof(unsigned int));
  STATIC_REQUIRE(thrust::detail::alignment_of<long int>::value == sizeof(long int));
  STATIC_REQUIRE(thrust::detail::alignment_of<unsigned long int>::value == sizeof(unsigned long int));
  STATIC_REQUIRE(thrust::detail::alignment_of<long long int>::value == sizeof(long long int));
  STATIC_REQUIRE(thrust::detail::alignment_of<unsigned long long int>::value == sizeof(unsigned long long int));
  STATIC_REQUIRE(thrust::detail::alignment_of<float>::value == sizeof(float));
  STATIC_REQUIRE(thrust::detail::alignment_of<double>::value == sizeof(double));
  STATIC_REQUIRE(thrust::detail::alignment_of<long double>::value == sizeof(long double));

  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_0>::value == sizeof(char));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_1>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_2>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_3>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_4>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_5>::value == sizeof(int));
  STATIC_REQUIRE(thrust::detail::alignment_of<alignof_mock_6>::value == sizeof(int));
}

template <std::size_t Align>
void test_aligned_type_instantiation()
{
  using type = typename thrust::detail::aligned_type<Align>::type;
  STATIC_REQUIRE(sizeof(type) >= 1lu);
  STATIC_REQUIRE(alignof(type) == Align);
  STATIC_REQUIRE(thrust::detail::alignment_of<type>::value == Align);
}

TEST_CASE("aligned_type", "[alignment]")
{
  test_aligned_type_instantiation<1>();
  test_aligned_type_instantiation<2>();
  test_aligned_type_instantiation<4>();
  test_aligned_type_instantiation<8>();
  test_aligned_type_instantiation<16>();
  test_aligned_type_instantiation<32>();
  test_aligned_type_instantiation<64>();
  test_aligned_type_instantiation<128>();
}

TEST_CASE("max_align_t", "[alignment]")
{
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(bool));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(signed char));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(unsigned char));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(char));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(short int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(unsigned short int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(unsigned int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(long int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(unsigned long int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(long long int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(unsigned long long int));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(float));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(double));
  STATIC_REQUIRE(alignof(thrust::detail::max_align_t) >= alignof(long double));
}

TEST_CASE("aligned_reinterpret_cast", "[alignment]")
{
  thrust::detail::aligned_type<1>* a1 = 0;
  thrust::detail::aligned_type<2>* a2 = 0;

  // Cast to type with stricter (larger) alignment requirement.
  a2 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<2>*>(a1);

  // Cast to type with less strict (smaller) alignment requirement.
  a1 = thrust::detail::aligned_reinterpret_cast<thrust::detail::aligned_type<1>*>(a2);
}
