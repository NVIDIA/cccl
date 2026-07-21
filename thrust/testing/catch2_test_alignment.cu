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

TEST_CASE("aligned_reinterpret_cast", "[alignment]")
{
  struct alignas(128) T1
  {
    char data_[128];
  };

  struct alignas(512) T2
  {
    char data_[512];
  };

  T1* a1 = nullptr;
  T2* a2 = nullptr;

  // Cast to type with stricter (larger) alignment requirement.
  a2 = thrust::detail::aligned_reinterpret_cast<T2*>(a1);

  // Cast to type with less strict (smaller) alignment requirement.
  a1 = thrust::detail::aligned_reinterpret_cast<T1*>(a2);
}
