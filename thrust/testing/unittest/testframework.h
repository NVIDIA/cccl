#pragma once

#include <thrust/detail/config.h>

#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/host_memory_resource.h>
#include <thrust/mr/universal_memory_resource.h>

#include <cuda/std/limits>

#include <cstdio>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "meta.h"
#include "util.h"

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

// A type that behaves as if it was a normal numeric type,
// so it can be used in the same tests as "normal" numeric types.
// NOTE: This is explicitly NOT proclaimed trivially reloctable.
class custom_numeric
{
public:
  _CCCL_HOST_DEVICE custom_numeric()
  {
    fill(0);
  }

  // Allow construction from any integral numeric.
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  _CCCL_HOST_DEVICE custom_numeric(const T& i)
  {
    fill(static_cast<int>(i));
  }

  _CCCL_HOST_DEVICE custom_numeric(const custom_numeric& other)
  {
    fill(other.value[0]);
  }

  _CCCL_HOST_DEVICE custom_numeric& operator=(int val)
  {
    fill(val);
    return *this;
  }

  _CCCL_HOST_DEVICE custom_numeric& operator=(const custom_numeric& other)
  {
    fill(other.value[0]);
    return *this;
  }

  // cast to void * instead of bool to fool overload resolution
  // WTB C++11 explicit conversion operators
  _CCCL_HOST_DEVICE operator void*() const
  {
    // static cast first to avoid MSVC warning C4312
    return reinterpret_cast<void*>(static_cast<std::size_t>(value[0]));
  }

#define DEFINE_OPERATOR(op)                               \
  _CCCL_HOST_DEVICE custom_numeric& operator op()         \
  {                                                       \
    fill(op value[0]);                                    \
    return *this;                                         \
  }                                                       \
  _CCCL_HOST_DEVICE custom_numeric operator op(int) const \
  {                                                       \
    custom_numeric ret(*this);                            \
    op ret;                                               \
    return ret;                                           \
  }

  DEFINE_OPERATOR(++)
  DEFINE_OPERATOR(--)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                            \
  _CCCL_HOST_DEVICE custom_numeric operator op() const \
  {                                                    \
    return custom_numeric(op value[0]);                \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(~)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                       \
  _CCCL_HOST_DEVICE custom_numeric operator op(const custom_numeric& other) const \
  {                                                                               \
    return custom_numeric(value[0] op other.value[0]);                            \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(*)
  DEFINE_OPERATOR(/)
  DEFINE_OPERATOR(%)
  DEFINE_OPERATOR(<<)
  DEFINE_OPERATOR(>>)
  DEFINE_OPERATOR(&)
  DEFINE_OPERATOR(|)
  DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define CONCAT(X, Y) X##Y

#define DEFINE_OPERATOR(op)                                                              \
  _CCCL_HOST_DEVICE custom_numeric& operator CONCAT(op, =)(const custom_numeric & other) \
  {                                                                                      \
    fill(value[0] op other.value[0]);                                                    \
    return *this;                                                                        \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(*)
  DEFINE_OPERATOR(/)
  DEFINE_OPERATOR(%)
  DEFINE_OPERATOR(<<)
  DEFINE_OPERATOR(>>)
  DEFINE_OPERATOR(&)
  DEFINE_OPERATOR(|)
  DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                                       \
  _CCCL_HOST_DEVICE friend bool operator op(const custom_numeric& lhs, const custom_numeric& rhs) \
  {                                                                                               \
    return lhs.value[0] op rhs.value[0];                                                          \
  }

  DEFINE_OPERATOR(==)
  DEFINE_OPERATOR(!=)
  DEFINE_OPERATOR(<)
  DEFINE_OPERATOR(<=)
  DEFINE_OPERATOR(>)
  DEFINE_OPERATOR(>=)
  DEFINE_OPERATOR(&&)
  DEFINE_OPERATOR(||)

#undef DEFINE_OPERATOR

  friend std::ostream& operator<<(std::ostream& os, const custom_numeric& val)
  {
    return os << "custom_numeric{" << val.value[0] << "}";
  }

private:
  int value[5];

  _CCCL_HOST_DEVICE void fill(int val)
  {
    for (int i = 0; i < 5; ++i)
    {
      value[i] = val;
    }
  }
};

namespace std
{
template <>
struct numeric_limits<custom_numeric> : numeric_limits<int>
{};
} // namespace std

_CCCL_BEGIN_NAMESPACE_CUDA_STD
template <>
struct numeric_limits<custom_numeric> : numeric_limits<int>
{};
_CCCL_END_NAMESPACE_CUDA_STD

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

inline void chop_prefix(std::string& str, const std::string& prefix)
{
  str.replace(str.find(prefix) == 0 ? 0 : str.size(), prefix.size(), "");
}

inline std::string base_class_name(const std::string& name)
{
  std::string result = name;

  // if the name begins with "struct ", chop it off
  chop_prefix(result, "struct ");

  // if the name begins with "class ", chop it off
  chop_prefix(result, "class ");

  const std::size_t first_lt = result.find_first_of("<");

  if (first_lt < result.size())
  {
    // chop everything including and after first "<"
    return result.replace(first_lt, result.size(), "");
  }
  else
  {
    return result;
  }
}

enum TestStatus
{
  Pass             = 0,
  Failure          = 1,
  KnownFailure     = 2,
  Error            = 3,
  UnknownException = 4
};

using ArgumentSet = std::set<std::string>;
using ArgumentMap = std::map<std::string, std::string>;

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

void set_test_sizes(const std::string&);

class UnitTest
{
public:
  std::string name;
  UnitTest() {}
  UnitTest(const char* name);
  virtual ~UnitTest() {}
  virtual void run() {}

  bool operator<(const UnitTest& u) const
  {
    return name < u.name;
  }
};

class UnitTestDriver;

class UnitTestDriver
{
  using TestMap = std::map<std::string, UnitTest*>;

  TestMap test_map;

  bool run_tests(std::vector<UnitTest*>& tests_to_run, const ArgumentMap& kwargs);

protected:
  // executed immediately after each test
  // \param test The UnitTest of interest
  // \param concise Whether or not to suppress output
  // \return true if all is well; false if the tests must be immediately aborted
  virtual bool post_test_smoke_check(const UnitTest& test, bool concise);

public:
  inline virtual ~UnitTestDriver() {}

  void register_test(UnitTest* test);
  virtual bool run_tests(const ArgumentSet& args, const ArgumentMap& kwargs);
  void list_tests();

  static UnitTestDriver& s_driver();
};

// Macro to create a single unittest
#define DECLARE_UNITTEST(TEST)           \
  class TEST##UnitTest : public UnitTest \
  {                                      \
  public:                                \
    TEST##UnitTest()                     \
        : UnitTest(#TEST)                \
    {}                                   \
    void run()                           \
    {                                    \
      TEST();                            \
    }                                    \
  };                                     \
  TEST##UnitTest TEST##Instance

#define DECLARE_UNITTEST_WITH_NAME(TEST, NAME) \
  class NAME##UnitTest : public UnitTest       \
  {                                            \
  public:                                      \
    NAME##UnitTest()                           \
        : UnitTest(#NAME)                      \
    {}                                         \
    void run()                                 \
    {                                          \
      TEST();                                  \
    }                                          \
  };                                           \
  NAME##UnitTest NAME##Instance

// Macro to create host and device versions of a
// unit test for a bunch of data types
#define DECLARE_VECTOR_UNITTEST(VTEST)                                                                                  \
  void VTEST##Host()                                                                                                    \
  {                                                                                                                     \
    VTEST<thrust::host_vector<signed char>>();                                                                          \
    VTEST<thrust::host_vector<short>>();                                                                                \
    VTEST<thrust::host_vector<int>>();                                                                                  \
    VTEST<thrust::host_vector<float>>();                                                                                \
    VTEST<thrust::host_vector<custom_numeric>>();                                                                       \
    /* MR vectors */                                                                                                    \
    VTEST<thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>>();     \
  }                                                                                                                     \
  void VTEST##Device()                                                                                                  \
  {                                                                                                                     \
    VTEST<thrust::device_vector<signed char>>();                                                                        \
    VTEST<thrust::device_vector<short>>();                                                                              \
    VTEST<thrust::device_vector<int>>();                                                                                \
    VTEST<thrust::device_vector<float>>();                                                                              \
    VTEST<thrust::device_vector<custom_numeric>>();                                                                     \
    /* MR vectors */                                                                                                    \
    VTEST<thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>>(); \
  }                                                                                                                     \
  void VTEST##Universal()                                                                                               \
  {                                                                                                                     \
    VTEST<thrust::universal_vector<int>>();                                                                             \
    VTEST<thrust::universal_host_pinned_vector<int>>();                                                                 \
  }                                                                                                                     \
  DECLARE_UNITTEST(VTEST##Host);                                                                                        \
  DECLARE_UNITTEST(VTEST##Device);                                                                                      \
  DECLARE_UNITTEST(VTEST##Universal);

// Same as above, but only for integral types
#define DECLARE_INTEGRAL_VECTOR_UNITTEST(VTEST)         \
  void VTEST##Host()                                    \
  {                                                     \
    VTEST<thrust::host_vector<signed char>>();          \
    VTEST<thrust::host_vector<short>>();                \
    VTEST<thrust::host_vector<int>>();                  \
  }                                                     \
  void VTEST##Device()                                  \
  {                                                     \
    VTEST<thrust::device_vector<signed char>>();        \
    VTEST<thrust::device_vector<short>>();              \
    VTEST<thrust::device_vector<int>>();                \
  }                                                     \
  void VTEST##Universal()                               \
  {                                                     \
    VTEST<thrust::universal_vector<int>>();             \
    VTEST<thrust::universal_host_pinned_vector<int>>(); \
  }                                                     \
  DECLARE_UNITTEST(VTEST##Host);                        \
  DECLARE_UNITTEST(VTEST##Device);                      \
  DECLARE_UNITTEST(VTEST##Universal);

// Macro to create instances of a test for several data types.
#define DECLARE_GENERIC_UNITTEST(TEST)   \
  class TEST##UnitTest : public UnitTest \
  {                                      \
  public:                                \
    TEST##UnitTest()                     \
        : UnitTest(#TEST)                \
    {}                                   \
    void run()                           \
    {                                    \
      TEST<signed char>();               \
      TEST<unsigned char>();             \
      TEST<short>();                     \
      TEST<unsigned short>();            \
      TEST<int>();                       \
      TEST<unsigned int>();              \
      TEST<float>();                     \
    }                                    \
  };                                     \
  TEST##UnitTest TEST##Instance

// Macro to create instances of a test for several array sizes.
#define DECLARE_SIZED_UNITTEST(TEST)                \
  class TEST##UnitTest : public UnitTest            \
  {                                                 \
  public:                                           \
    TEST##UnitTest()                                \
        : UnitTest(#TEST)                           \
    {}                                              \
    void run()                                      \
    {                                               \
      std::vector<size_t> sizes = get_test_sizes(); \
      for (size_t i = 0; i != sizes.size(); ++i)    \
      {                                             \
        TEST(sizes[i]);                             \
      }                                             \
    }                                               \
  };                                                \
  TEST##UnitTest TEST##Instance

// Macro to create instances of a test for several data types and array sizes
#define DECLARE_VARIABLE_UNITTEST(TEST)             \
  class TEST##UnitTest : public UnitTest            \
  {                                                 \
  public:                                           \
    TEST##UnitTest()                                \
        : UnitTest(#TEST)                           \
    {}                                              \
    void run()                                      \
    {                                               \
      std::vector<size_t> sizes = get_test_sizes(); \
      for (size_t i = 0; i != sizes.size(); ++i)    \
      {                                             \
        TEST<signed char>(sizes[i]);                \
        TEST<unsigned char>(sizes[i]);              \
        TEST<short>(sizes[i]);                      \
        TEST<unsigned short>(sizes[i]);             \
        TEST<int>(sizes[i]);                        \
        TEST<unsigned int>(sizes[i]);               \
        TEST<float>(sizes[i]);                      \
        TEST<double>(sizes[i]);                     \
      }                                             \
    }                                               \
  };                                                \
  TEST##UnitTest TEST##Instance

#define DECLARE_INTEGRAL_VARIABLE_UNITTEST(TEST)    \
  class TEST##UnitTest : public UnitTest            \
  {                                                 \
  public:                                           \
    TEST##UnitTest()                                \
        : UnitTest(#TEST)                           \
    {}                                              \
    void run()                                      \
    {                                               \
      std::vector<size_t> sizes = get_test_sizes(); \
      for (size_t i = 0; i != sizes.size(); ++i)    \
      {                                             \
        TEST<signed char>(sizes[i]);                \
        TEST<unsigned char>(sizes[i]);              \
        TEST<short>(sizes[i]);                      \
        TEST<unsigned short>(sizes[i]);             \
        TEST<int>(sizes[i]);                        \
        TEST<unsigned int>(sizes[i]);               \
      }                                             \
    }                                               \
  };                                                \
  TEST##UnitTest TEST##Instance

#define DECLARE_GENERIC_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPES, NAME) \
  ::SimpleUnitTest<TEST, TYPES> NAME##_instance(#NAME) /**/

#define DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES_AND_NAME(TEST, TYPES, NAME) \
  ::VariableUnitTest<TEST, TYPES> NAME##_instance(#NAME) /**/

#define DECLARE_GENERIC_UNITTEST_WITH_TYPES(TEST, TYPES) ::SimpleUnitTest<TEST, TYPES> TEST##_instance(#TEST) /**/

#define DECLARE_GENERIC_SIZED_UNITTEST_WITH_TYPES(TEST, TYPES) \
  ::VariableUnitTest<TEST, TYPES> TEST##_instance(#TEST) /**/

template <template <typename> class TestName, typename TypeList>
class SimpleUnitTest : public UnitTest
{
public:
  SimpleUnitTest()
      : UnitTest(base_class_name(unittest::type_name<TestName<int>>()).c_str())
  {}

  SimpleUnitTest(const char* name)
      : UnitTest(name)
  {}

  void run()
  {
    // get the first type in the list
    using first_type = typename unittest::get_type<TypeList, 0>::type;

    unittest::for_each_type<TypeList, TestName, first_type, 0> for_each;

    // loop over the types
    for_each();
  }
}; // end SimpleUnitTest

template <template <typename> class TestName, typename TypeList>
class VariableUnitTest : public UnitTest
{
public:
  VariableUnitTest()
      : UnitTest(base_class_name(unittest::type_name<TestName<int>>()).c_str())
  {}

  VariableUnitTest(const char* name)
      : UnitTest(name)
  {}

  void run()
  {
    std::vector<size_t> sizes = get_test_sizes();
    for (size_t i = 0; i != sizes.size(); ++i)
    {
      // get the first type in the list
      using first_type = typename unittest::get_type<TypeList, 0>::type;

      unittest::for_each_type<TypeList, TestName, first_type, 0> loop;

      // loop over the types
      loop(sizes[i]);
    }
  }
}; // end VariableUnitTest

template <template <typename> class TestName,
          typename TypeList,
          template <typename, typename> class Vector,
          template <typename> class Alloc>
struct VectorUnitTest : public UnitTest
{
  VectorUnitTest()
      : UnitTest((base_class_name(unittest::type_name<TestName<Vector<int, Alloc<int>>>>()) + "<"
                  + base_class_name(unittest::type_name<Vector<int, Alloc<int>>>()) + ">")
                   .c_str())
  {}

  VectorUnitTest(const char* name)
      : UnitTest(name)
  {}

  void run()
  {
    // zip up the type list with Alloc
    using AllocList = typename unittest::transform1<TypeList, Alloc>::type;

    // zip up the type list & alloc list with Vector
    using VectorList = typename unittest::transform2<TypeList, AllocList, Vector>::type;

    // get the first type in the list
    using first_type = typename unittest::get_type<VectorList, 0>::type;

    unittest::for_each_type<VectorList, TestName, first_type, 0> loop;

    // loop over the types
    loop(0);
  }
}; // end VectorUnitTest
