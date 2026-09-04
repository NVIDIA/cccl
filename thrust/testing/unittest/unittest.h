#pragma once

// Here was Thrust's legacy unit test framework. It's core was replaced by Catch2, but the APIs remain for all tests
// that have not been migrated yet.

#include <thrust/detail/type_traits.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/mr/device_memory_resource.h>
#include <thrust/mr/host_memory_resource.h>
#include <thrust/random.h>
#include <thrust/universal_vector.h>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstdlib>
#include <iosfwd>
#include <limits>
#include <string>
#include <typeinfo>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>

#ifdef __GNUC__
#  include <cxxabi.h>
#endif // __GNUC__

// workaround for error #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
#if _CCCL_COMPILER(NVHPC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION _Pragma("diag push")
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION  _Pragma("diag pop")
#endif
// workaround for error
// * MSVC14.39: #3185-D: no '#pragma diagnostic push' was found to match this 'diagnostic pop'
// * MSVC14.29: internal error: assertion failed: alloc_copy_of_pending_pragma: copied pragma has source sequence entry
//              (pragma.c, line 526 in alloc_copy_of_pending_pragma)
// see also upstream Catch2 issue: https://github.com/catchorg/Catch2/issues/2636
#if _CCCL_COMPILER(MSVC)
#  undef CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  undef CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#  define CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
#  define CATCH_INTERNAL_SUPPRESS_UNUSED_VARIABLE_WARNINGS
#endif

// ==== merged from the former unittest/detail/special_types.h ====

template <typename T, unsigned int N>
struct FixedVector
{
  T data[N];

  _CCCL_HOST_DEVICE FixedVector()
  {
    for (unsigned int i = 0; i < N; i++)
    {
      data[i] = T();
    }
  }

  _CCCL_HOST_DEVICE explicit FixedVector(T init)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      data[i] = init;
    }
  }

  _CCCL_HOST_DEVICE
#if _CCCL_COMPILER(NVHPC)
  __attribute__((noinline))
#endif
  FixedVector operator+(const FixedVector& bs) const
  {
    FixedVector output;
    for (unsigned int i = 0; i < N; i++)
    {
      output.data[i] = data[i] + bs.data[i];
    }
    return output;
  }

  _CCCL_HOST_DEVICE bool operator<(const FixedVector& bs) const
  {
    for (unsigned int i = 0; i < N; i++)
    {
      if (data[i] < bs.data[i])
      {
        return true;
      }
      if (bs.data[i] < data[i])
      {
        return false;
      }
    }
    return false;
  }

  _CCCL_HOST_DEVICE bool operator==(const FixedVector& bs) const
  {
    for (unsigned int i = 0; i < N; i++)
    {
      if (!(data[i] == bs.data[i]))
      {
        return false;
      }
    }
    return true;
  }
};

template <typename Key, typename Value>
struct key_value
{
  using key_type   = Key;
  using value_type = Value;

  _CCCL_HOST_DEVICE key_value()
      : key()
      , value()
  {}

  _CCCL_HOST_DEVICE key_value(key_type k, value_type v)
      : key(k)
      , value(v)
  {}

  _CCCL_HOST_DEVICE bool operator<(const key_value& rhs) const
  {
    return key < rhs.key;
  }

  _CCCL_HOST_DEVICE bool operator>(const key_value& rhs) const
  {
    return key > rhs.key;
  }

  _CCCL_HOST_DEVICE bool operator==(const key_value& rhs) const
  {
    return key == rhs.key && value == rhs.value;
  }

  _CCCL_HOST_DEVICE bool operator!=(const key_value& rhs) const
  {
    return !(*this == rhs);
  }

  friend std::ostream& operator<<(std::ostream& os, const key_value& kv)
  {
    return os << "(" << kv.key << ", " << kv.value << ")";
  }

  key_type key;
  value_type value;
};

struct user_swappable
{
  _CCCL_HOST_DEVICE user_swappable(bool swapped = false)
      : was_swapped(swapped)
  {}

  bool was_swapped;

  friend _CCCL_HOST_DEVICE bool operator==(const user_swappable& x, const user_swappable& y)
  {
    return x.was_swapped == y.was_swapped;
  }

  friend _CCCL_HOST_DEVICE void swap(user_swappable& x, user_swappable& y) noexcept
  {
    x.was_swapped = true;
    y.was_swapped = false;
  }
};

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
  _CCCL_HOST_DEVICE explicit operator bool() const
  {
    return value[0] != 0;
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

#define DEFINE_OPERATOR(op)                                                             \
  _CCCL_HOST_DEVICE custom_numeric& operator CONCAT(op, =)(const custom_numeric& other) \
  {                                                                                     \
    fill(value[0] op other.value[0]);                                                   \
    return *this;                                                                       \
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
#undef CONCAT

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

// Inheriting from classes in anonymous namespaces is not allowed.
// The anonymous namespace tests don't use these, so just disable them:
#ifndef THRUST_USE_ANON_NAMESPACE

struct my_system : THRUST_NS_QUALIFIER::device_execution_policy<my_system>
{
  my_system(int) {}

  my_system(const my_system& other)
      : num_copies(other.num_copies + 1)
  {}

  void validate_dispatch()
  {
    correctly_dispatched = (num_copies == 0);
  }

  bool is_valid() const
  {
    return correctly_dispatched;
  }

private:
  bool correctly_dispatched = false;

  // count the number of copies so that we can validate
  // that dispatch does not introduce any
  unsigned int num_copies = 0;
};

struct my_tag : THRUST_NS_QUALIFIER::device_execution_policy<my_tag>
{};

#endif // THRUST_USE_ANON_NAMESPACE

namespace unittest
{
using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::int8_t;

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;
} // namespace unittest

// ==== merged from the former unittest/detail/random.h ====

namespace unittest
{
namespace detail
{
inline unsigned int hash(unsigned int a)
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}
} // namespace detail

template <typename T>
struct generate_random_integer
{
  T operator()(unsigned int i) const
  {
    THRUST_NS_QUALIFIER::default_random_engine rng(detail::hash(i));
    if constexpr (::cuda::std::is_same_v<T, bool>)
    {
      THRUST_NS_QUALIFIER::uniform_int_distribution<unsigned int> dist(0, 1);
      return dist(rng) == 1;
    }
    else if constexpr (::cuda::std::is_integral_v<T>)
    {
      T const min = ::cuda::std::numeric_limits<T>::min();
      T const max = ::cuda::std::numeric_limits<T>::max();
      THRUST_NS_QUALIFIER::uniform_int_distribution<T> dist(min, max);
      return static_cast<T>(dist(rng));
    }
    else if constexpr (::cuda::std::is_floating_point_v<T>)
    {
      T const min = ::cuda::std::numeric_limits<T>::lowest();
      T const max = ::cuda::std::numeric_limits<T>::max();
      THRUST_NS_QUALIFIER::uniform_real_distribution<T> dist(min, max);
      return static_cast<T>(dist(rng));
    }
    else
    {
      return static_cast<T>(rng());
    }
  }
};

template <typename T>
struct generate_random_sample
{
  T operator()(unsigned int i) const
  {
    THRUST_NS_QUALIFIER::default_random_engine rng(detail::hash(i));
    THRUST_NS_QUALIFIER::uniform_int_distribution<unsigned int> dist(0, 20);

    return static_cast<T>(dist(rng));
  }
};

template <typename T>
THRUST_NS_QUALIFIER::host_vector<T> random_integers(const size_t N)
{
  THRUST_NS_QUALIFIER::host_vector<T> vec(N);
  THRUST_NS_QUALIFIER::transform(
    THRUST_NS_QUALIFIER::counting_iterator{0u},
    THRUST_NS_QUALIFIER::counting_iterator{static_cast<unsigned int>(N)},
    vec.begin(),
    generate_random_integer<T>());

  return vec;
}

template <typename T>
T random_integer()
{
  return generate_random_integer<T>()(0);
}

template <typename T>
THRUST_NS_QUALIFIER::host_vector<T> random_samples(const size_t N)
{
  THRUST_NS_QUALIFIER::host_vector<T> vec(N);
  THRUST_NS_QUALIFIER::transform(
    THRUST_NS_QUALIFIER::counting_iterator{0u},
    THRUST_NS_QUALIFIER::counting_iterator{static_cast<unsigned int>(N)},
    vec.begin(),
    generate_random_sample<T>());

  return vec;
}
}; // end namespace unittest

// ==== shared Catch2 type lists (merged from the former catch2_test_helper.h) ====

// corresponds to DECLARE_VECTOR_UNITTEST
using vector_list = cuda::std::__type_list<
  // host
  thrust::host_vector<signed char>,
  thrust::host_vector<short>,
  thrust::host_vector<int>,
  thrust::host_vector<float>,
  thrust::host_vector<custom_numeric>,
  thrust::host_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::host_memory_resource>>,
  // device
  thrust::device_vector<signed char>,
  thrust::device_vector<short>,
  thrust::device_vector<int>,
  thrust::device_vector<float>,
  thrust::device_vector<custom_numeric>,
  thrust::device_vector<int, thrust::mr::stateless_resource_allocator<int, thrust::device_memory_resource>>,
  // universal
  thrust::universal_vector<int>,
  thrust::universal_host_pinned_vector<int>>;

// corresponds to DECLARE_INTEGRAL_VECTOR_UNITTEST
using integral_vector_list = cuda::std::__type_list<
  // host
  thrust::host_vector<signed char>,
  thrust::host_vector<short>,
  thrust::host_vector<int>,
  // device
  thrust::device_vector<signed char>,
  thrust::device_vector<short>,
  thrust::device_vector<int>,
  // universal
  thrust::universal_vector<int>,
  thrust::universal_host_pinned_vector<int>>;

// corresponds to DECLARE_GENERIC_UNITTEST
using generic_list =
  cuda::std::__type_list<signed char, unsigned char, short, unsigned short, int, unsigned int, float>;

// corresponds to DECLARE_VARIABLE_UNITTEST
using variable_list =
  cuda::std::__type_list<signed char, unsigned char, short, unsigned short, int, unsigned int, float, double>;

// gcc >= 11 emits bogus -Werror=stringop-overflow diagnostics ("writing N bytes into a region of size 0") for copies
// of small vectors of narrow types. The culprit optimizations are the tree vectorizer and the loop-distribute-patterns
// pass (which rewrites copy loops into memmove). Disabling both on the affected test functions works around it.
#if _CCCL_COMPILER(GCC, >=, 11)
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER \
    __attribute__((optimize("no-tree-vectorize", "no-tree-loop-distribute-patterns")))
#else
#  define THRUST_DISABLE_BROKEN_GCC_VECTORIZER
#endif

namespace unittest::detail
{
template <class T, class = void>
struct has_value_type : ::cuda::std::false_type
{};
template <class T>
struct has_value_type<T, ::cuda::std::void_t<typename T::value_type>> : ::cuda::std::true_type
{};

// device_vector: copy to host first
template <class T, class Alloc>
std::vector<T> to_vec(thrust::device_vector<T, Alloc> const& vec)
{
  thrust::host_vector<T> temp = vec;
  return std::vector<T>{temp.begin(), temp.end()};
}

// any other host-accessible container (host_vector, universal_vector, std::vector, vector_base, ...)
template <class C, ::cuda::std::enable_if_t<has_value_type<C>::value, int> = 0>
std::vector<typename C::value_type> to_vec(C const& c)
{
  return std::vector<typename C::value_type>{c.begin(), c.end()};
}

// scalar: wrap into a one-element vector
template <class T, ::cuda::std::enable_if_t<!has_value_type<T>::value, int> = 0>
std::vector<T> to_vec(T const& x)
{
  return std::vector<T>{x};
}

// Catch2's Approx matcher only understands arithmetic element types. For complex numbers we flatten the vector into
// its interleaved real/imaginary components so the matcher can compare them component-wise.
template <class T>
std::vector<T> to_approx(std::vector<T> v)
{
  return v;
}

template <template <class> class Complex, class T>
std::vector<T> to_approx(std::vector<Complex<T>> const& v)
{
  std::vector<T> out;
  out.reserve(2 * v.size());
  for (auto const& z : v)
  {
    out.push_back(z.real());
    out.push_back(z.imag());
  }
  return out;
}
} // namespace unittest::detail

#define ASSERT_EQUAL(X, Y)     REQUIRE((X) == (Y))
#define ASSERT_NOT_EQUAL(X, Y) REQUIRE((X) != (Y))
// The QUIET variants wrap the whole comparison in an extra pair of parentheses so that Catch2 does not decompose the
// expression. This avoids stringifying the operands, which is required for types that are not streamable (e.g. vectors
// of tuples or other types without an ostream operator<<).
#define ASSERT_EQUAL_QUIET(X, Y)     REQUIRE((X == Y))
#define ASSERT_NOT_EQUAL_QUIET(X, Y) REQUIRE((X != Y))
#define ASSERT_LEQUAL(X, Y)          REQUIRE((X) <= (Y))
#define ASSERT_GEQUAL(X, Y)          REQUIRE((X) >= (Y))
#define ASSERT_LESS(X, Y)            REQUIRE((X) < (Y))
#define ASSERT_GREATER(X, Y)         REQUIRE((X) > (Y))
#define ASSERT_ALMOST_EQUAL(X, Y)                                                \
  {                                                                              \
    auto vec_ref = ::unittest::detail::to_approx(::unittest::detail::to_vec(X)); \
    auto vec_out = ::unittest::detail::to_approx(::unittest::detail::to_vec(Y)); \
    REQUIRE_THAT(vec_ref, Catch::Matchers::Approx(vec_out));                     \
  }

#define ASSERT_THROWS(EXPR, EXCEPTION_TYPE) CHECK_THROWS_AS(EXPR, EXCEPTION_TYPE)
#define KNOWN_FAILURE                       FAIL()

namespace unittest
{
template <typename... Ts>
using type_list = ::cuda::std::__type_list<Ts...>;
} // namespace unittest

// define some common lists of types
using ThirtyTwoBitTypes = unittest::type_list<int, unsigned int, float>;

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

using SmallIntegralTypes = unittest::type_list<char, signed char, unsigned char, short, unsigned short>;

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

namespace unittest
{
inline std::string demangle(const char* name)
{
  // for demangling the result of type_info.name() with msvc, type_info.name() is already demangled
#if __GNUC__ && !_NVHPC_CUDA
  int status     = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);
  return result;
#else
  return name;
#endif
}

template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
} // end type_name()

// Use this with counting_iterator to avoid generating a range larger than we can represent.
// TODO: This probably won't work for `half`.
template <typename T>
T truncate_to_max_representable(std::size_t n)
{
  if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    return ::cuda::std::min<T>(static_cast<T>(n), ::cuda::std::numeric_limits<T>::max());
  }
  else
  {
    return static_cast<T>(
      ::cuda::std::min<std::size_t>(n, static_cast<std::size_t>(::cuda::std::numeric_limits<T>::max())));
  }
}
} // namespace unittest
