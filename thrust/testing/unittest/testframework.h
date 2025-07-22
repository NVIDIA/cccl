#pragma once

#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/limits>

#include <cstdio>
#include <limits>
#include <vector>

#include "catch2_test_helper.h"
#include "special_types.h"
#include "util.h"

namespace unittest
{
template <typename... Ts>
using type_list = ::cuda::std::__type_list<Ts...>;
} // namespace unittest

// define some common lists of types
using ThirtyTwoBitTypes = unittest::type_list<int, unsigned int, float>;

using SixtyFourBitTypes = unittest::type_list<long long, unsigned long long, double>;

using IntegralTypes = unittest::type_list<
  char,
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned int,
  long,
  unsigned long,
  long long,
  unsigned long long>;

using SignedIntegralTypes = unittest::type_list<signed char, short, int, long, long long>;

using UnsignedIntegralTypes =
  unittest::type_list<unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long>;

using ByteTypes = unittest::type_list<char, signed char, unsigned char>;

using SmallIntegralTypes = unittest::type_list<char, signed char, unsigned char, short, unsigned short>;

using LargeIntegralTypes = unittest::type_list<long long, unsigned long long>;

using FloatingPointTypes = unittest::type_list<float, double>;

using NumericTypes = unittest::type_list<
  char,
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned int,
  long,
  unsigned long,
  long long,
  unsigned long long,
  float,
  double,
  custom_numeric>;

using BuiltinNumericTypes = unittest::type_list<
  char,
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned int,
  long,
  unsigned long,
  long long,
  unsigned long long,
  float,
  double>;

// clang-format off
inline constexpr size_t standard_test_sizes[] =
{
  0, 1, 2, 3, 4, 5, 8, 10, 13, 16, 17, 19, 27, 30, 31, 32,
  33, 35, 42, 53, 58, 63, 64, 65, 72, 97, 100, 127, 128, 129, 142, 183, 192, 201, 240, 255, 256,
  257, 302, 511, 512, 513, 687, 900, 1023, 1024, 1025, 1565, 1786, 1973, 2047, 2048, 2049, 3050, 4095, 4096,
  4097, 5030, 7791, 10000, 10027, 12345, 16384, 17354, 26255, 32768, 43718, 65533, 65536,
  65539, 123456, 131072, 731588, 1048575, 1048576,
  3398570, 9760840, (1 << 24) - 1, (1 << 24),
  (1 << 24) + 1, (1 << 25) - 1, (1 << 25), (1 << 25) + 1, (1 << 26) - 1, 1 << 26,
  (1 << 26) + 1, (1 << 27) - 1, (1 << 27)
};
// clang-format on

inline constexpr size_t tiny_threshold    = 1 << 5; //   32
inline constexpr size_t small_threshold   = 1 << 8; //  256
inline constexpr size_t medium_threshold  = 1 << 12; //   4K
inline constexpr size_t default_threshold = 1 << 16; //  64K
inline constexpr size_t large_threshold   = 1 << 20; //   1M
inline constexpr size_t huge_threshold    = 1 << 24; //  16M
inline constexpr size_t epic_threshold    = 1 << 26; //  64M
inline constexpr size_t max_threshold     = (std::numeric_limits<size_t>::max)();

inline std::vector<size_t> test_sizes = [] {
  std::vector<size_t> v;
  for (size_t s : standard_test_sizes)
  {
    if (s <= default_threshold)
    {
      v.push_back(s);
    }
  }
  return v;
}();

inline const std::vector<size_t>& get_test_sizes()
{
  return test_sizes;
}

// Macro to create a single unittest
#define DECLARE_UNITTEST(TEST)                    \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    TEST();                                       \
  }

#define DECLARE_UNITTEST_WITH_NAME(TEST, NAME)    \
  TEST_CASE(#NAME, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    TEST();                                       \
  }

// Macro to create host and device versions of a
// unit test for a bunch of data types
#define DECLARE_VECTOR_UNITTEST(VTEST)                                                                                  \
  TEST_CASE(#VTEST, THRUST_PP_STRINGIZE(__FILE__))                                                                      \
  {                                                                                                                     \
    /* host */                                                                                                          \
    VTEST<thrust::host_vector<signed char>>();                                                                          \
    VTEST<thrust::host_vector<short>>();                                                                                \
    VTEST<thrust::host_vector<int>>();                                                                                  \
    VTEST<thrust::host_vector<float>>();                                                                                \
    VTEST<thrust::host_vector<custom_numeric>>();                                                                       \
    VTEST<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>();     \
    /* device */                                                                                                        \
    VTEST<thrust::device_vector<signed char>>();                                                                        \
    VTEST<thrust::device_vector<short>>();                                                                              \
    VTEST<thrust::device_vector<int>>();                                                                                \
    VTEST<thrust::device_vector<float>>();                                                                              \
    VTEST<thrust::device_vector<custom_numeric>>();                                                                     \
    VTEST<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>(); \
    /* universal*/                                                                                                      \
    VTEST<thrust::universal_vector<int>>();                                                                             \
    VTEST<thrust::universal_host_pinned_vector<int>>();                                                                 \
  }

// Same as above, but only for integral types
#define DECLARE_INTEGRAL_VECTOR_UNITTEST(VTEST)         \
  void VTEST##Host()                                    \
  {                                                     \
    /* host */                                          \
    VTEST<thrust::host_vector<signed char>>();          \
    VTEST<thrust::host_vector<short>>();                \
    VTEST<thrust::host_vector<int>>();                  \
    /* device */                                        \
    VTEST<thrust::device_vector<signed char>>();        \
    VTEST<thrust::device_vector<short>>();              \
    VTEST<thrust::device_vector<int>>();                \
    /* universal*/                                      \
    VTEST<thrust::universal_vector<int>>();             \
    VTEST<thrust::universal_host_pinned_vector<int>>(); \
  }

// Macro to create instances of a test for several data types.
#define DECLARE_GENERIC_UNITTEST(TEST)            \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    TEST<signed char>();                          \
    TEST<unsigned char>();                        \
    TEST<short>();                                \
    TEST<unsigned short>();                       \
    TEST<int>();                                  \
    TEST<unsigned int>();                         \
    TEST<float>();                                \
  }

// Macro to create instances of a test for several array sizes.
#define DECLARE_SIZED_UNITTEST(TEST)              \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    for (size_t s : get_test_sizes())             \
    {                                             \
      TEST(s);                                    \
    }                                             \
  }

namespace unittest::detail
{
template <template <typename> typename TestFunc, template <typename...> typename L, typename... Ts, typename... Args>
void for_each_type(L<Ts...>, Args&&... args)
{
  (..., TestFunc<Ts>{}(::cuda::std::forward<Args>(args)...));
}
} // namespace unittest::detail

#define DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TEST, ...)   \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__))              \
  {                                                            \
    for (size_t s : get_test_sizes())                          \
    {                                                          \
      unittest::detail::for_each_type<TEST>(__VA_ARGS__{}, s); \
    }                                                          \
  }

// Macro to create instances of a test for several data types and array sizes
#define DECLARE_VARIABLE_UNITTEST(TEST)           \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    for (size_t s : get_test_sizes())             \
    {                                             \
      TEST<signed char>(s);                       \
      TEST<unsigned char>(s);                     \
      TEST<short>(s);                             \
      TEST<unsigned short>(s);                    \
      TEST<int>(s);                               \
      TEST<unsigned int>(s);                      \
      TEST<float>(s);                             \
      TEST<double>(s);                            \
    }                                             \
  }

#define DECLARE_INTEGRAL_VARIABLE_UNITTEST(TEST)  \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__)) \
  {                                               \
    for (size_t s : get_test_sizes())             \
    {                                             \
      TEST<signed char>(s);                       \
      TEST<unsigned char>(s);                     \
      TEST<short>(s);                             \
      TEST<unsigned short>(s);                    \
      TEST<int>(s);                               \
      TEST<unsigned int>(s);                      \
    }                                             \
  }

namespace unittest::detail
{
template <template <typename> typename TestFunc,
          template <typename...> typename Vector,
          template <typename> typename Alloc,
          template <typename...> typename L,
          typename... Ts>
void invoke_vector_unittest(L<Ts...>)
{
  (..., TestFunc<Vector<Ts, Alloc<Ts>>>{}(0));
}
} // namespace unittest::detail

#define DECLARE_VECTOR_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPES, VECTOR, ALLOC, NAME) \
  TEST_CASE(#NAME, THRUST_PP_STRINGIZE(__FILE__))                                     \
  {                                                                                   \
    unittest::detail::invoke_vector_unittest<TEST, VECTOR, ALLOC>(TYPES{});           \
  }

#define DECLARE_GENERIC_UNITTEST_WITH_TYPES(TEST, ...)    \
  TEST_CASE(#TEST, THRUST_PP_STRINGIZE(__FILE__))         \
  {                                                       \
    unittest::detail::for_each_type<TEST>(__VA_ARGS__{}); \
  }

#define DECLARE_VECTOR_UNITTEST_WITH_TYPES(TEST, TYPES, VECTOR, ALLOC) \
  DECLARE_VECTOR_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPEWS, VECTOR, ALLOC, TEST)
