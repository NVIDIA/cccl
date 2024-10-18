#include <cuda/__functional/address_stability.h>

#include <unittest/unittest.h>

struct addable
{
  _CCCL_HOST_DEVICE friend auto operator+(const addable&, const addable&) -> addable
  {
    return addable{};
  }
};

void TestAddressStabilityLibcuxx()
{
  using ::cuda::allow_copied_arguments;
  using ::cuda::allows_copied_arguments;
  using ::cuda::callable_allowing_copied_arguments;

  // libcu++ function objects with known types
  static_assert(allows_copied_arguments<::cuda::std::plus<int>, int, int>::value, "");
  static_assert(allows_copied_arguments<::cuda::std::plus<>, int, int>::value, "");

  // libcu++ function objects with unknown types
  static_assert(!allows_copied_arguments<::cuda::std::plus<addable>, addable, addable>::value, "");
  static_assert(!allows_copied_arguments<::cuda::std::plus<>, addable, addable>::value, "");

  // libcu++ function objects with unknown types and opt-in
  static_assert(
    allows_copied_arguments<decltype(allow_copied_arguments(cuda::std::plus<addable>{})), addable, addable>::value, "");
  static_assert(
    allows_copied_arguments<callable_allowing_copied_arguments<::cuda::std::plus<addable>>, addable, addable>::value,
    "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(cuda::std::plus<>{})), addable, addable>::value,
                "");
  static_assert(
    allows_copied_arguments<callable_allowing_copied_arguments<::cuda::std::plus<>>, addable, addable>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityLibcuxx);

void TestAddressStabilityThrust()
{
  using ::cuda::allow_copied_arguments;
  using ::cuda::allows_copied_arguments;
  using ::cuda::callable_allowing_copied_arguments;

  // thrust function objects with known types
  static_assert(allows_copied_arguments<thrust::plus<int>, int, int>::value, "");
  static_assert(allows_copied_arguments<thrust::plus<>, int, int>::value, "");

  // thrust function objects with unknown types
  static_assert(!allows_copied_arguments<thrust::plus<addable>, addable, addable>::value, "");
  static_assert(!allows_copied_arguments<thrust::plus<>, addable, addable>::value, "");

  // thrust function objects with unknown types and opt-in
  static_assert(
    allows_copied_arguments<decltype(allow_copied_arguments(thrust::plus<addable>{})), addable, addable>::value, "");
  static_assert(
    allows_copied_arguments<callable_allowing_copied_arguments<thrust::plus<addable>>, addable, addable>::value, "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(thrust::plus<>{})), addable, addable>::value,
                "");
  static_assert(allows_copied_arguments<callable_allowing_copied_arguments<thrust::plus<>>, addable, addable>::value,
                "");
}
DECLARE_UNITTEST(TestAddressStabilityThrust);

template <typename T>
struct my_plus
{
  _CCCL_HOST_DEVICE auto operator()(T a, T b) const -> T
  {
    return a + b;
  }
};

struct overloaded
{
  // can copy
  _CCCL_HOST_DEVICE auto operator()(int a, int b) const -> int
  {
    return a + b;
  }

  // cannot copy
  _CCCL_HOST_DEVICE auto operator()(const float& a, const float& b) const -> float
  {
    return a + b;
  }
};

#define MY_GENERIC_PLUS(suffix, param)                             \
  struct my_generic_plus##suffix                                   \
  {                                                                \
    template <typename T>                                          \
    _CCCL_HOST_DEVICE auto operator()(param a, param b) const -> T \
    {                                                              \
      return a + b;                                                \
    }                                                              \
  }

MY_GENERIC_PLUS(, T);
MY_GENERIC_PLUS(_lref, T&);
MY_GENERIC_PLUS(_rref, T&&);
MY_GENERIC_PLUS(_clref, const T&);
MY_GENERIC_PLUS(_crref, const T&&);

void TestAddressStabilityUserDefinedFunctionObject()
{
  using ::cuda::allow_copied_arguments;
  using ::cuda::allows_copied_arguments;
  using ::cuda::callable_allowing_copied_arguments;

  // by-value overload
  static_assert(allows_copied_arguments<my_plus<int>, int, int>::value, "");

  // by-reference overload
  static_assert(!allows_copied_arguments<my_plus<int&>, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_plus<const int&>, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_plus<int&&>, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_plus<const int&&>, int, int>::value, "");

  // by-reference overload with opt-in
  static_assert(allows_copied_arguments<callable_allowing_copied_arguments<my_plus<int&>>, int, int>::value, "");
  static_assert(allows_copied_arguments<callable_allowing_copied_arguments<my_plus<const int&>>, int, int>::value, "");
  static_assert(allows_copied_arguments<callable_allowing_copied_arguments<my_plus<int&&>>, int, int>::value, "");
  static_assert(allows_copied_arguments<callable_allowing_copied_arguments<my_plus<const int&&>>, int, int>::value, "");

  // overloaded set
  static_assert(!allows_copied_arguments<overloaded, int, int>::value, ""); // may be solvable
  static_assert(!allows_copied_arguments<overloaded, float, float>::value, "");

  // call operator template
  static_assert(!allows_copied_arguments<my_generic_plus, int, int>::value, ""); // may be solvable
  static_assert(!allows_copied_arguments<my_generic_plus_lref, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_generic_plus_rref, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_generic_plus_clref, int, int>::value, "");
  static_assert(!allows_copied_arguments<my_generic_plus_crref, int, int>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityUserDefinedFunctionObject);

_CCCL_HOST_DEVICE auto my_plus_func(int a, int b) -> int
{
  return a + b;
}

_CCCL_HOST_DEVICE auto my_plus_func_ref(const int& a, const int& b) -> int
{
  return a + b;
}

void TestAddressStabilityUserDefinedFunctions()
{
  using ::cuda::allow_copied_arguments;
  using ::cuda::allows_copied_arguments;
  using ::cuda::callable_allowing_copied_arguments;

  // user-defined function types
  static_assert(allows_copied_arguments<int(int, int), int, int>::value, "");
  static_assert(!allows_copied_arguments<int(const int&, const int&), int, int>::value, "");

  static_assert(allows_copied_arguments<decltype(my_plus_func), int, int>::value, "");
  static_assert(!allows_copied_arguments<decltype(my_plus_func_ref), int, int>::value, "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(my_plus_func_ref)), int, int>::value, "");

  // user-defined function pointer types
  static_assert(allows_copied_arguments<int (*)(int, int), int, int>::value, "");
  static_assert(!allows_copied_arguments<int (*)(const int&, const int&), int, int>::value, "");

  static_assert(allows_copied_arguments<decltype(&my_plus_func), int, int>::value, "");
  static_assert(!allows_copied_arguments<decltype(&my_plus_func_ref), int, int>::value, "");
  static_assert(allows_copied_arguments<decltype(allow_copied_arguments(&my_plus_func_ref)), int, int>::value, "");

  // TODO(bgruber): test user-defined function reference types?
}
DECLARE_UNITTEST(TestAddressStabilityUserDefinedFunctions);

struct my_plus_proxy_ref
{
  int* a_ptr;

  // has a by-value argument
  auto operator()(::cuda::std::reference_wrapper<int> a, ::cuda::std::reference_wrapper<int> b) const -> int
  {
    int* address = &a.get(); // allows to recover the address of the argument
    ASSERT_EQUAL(address, a_ptr);
    return a + b;
  }
};

void TestAddressStabilityProxyReferences()
{
  using ::cuda::allows_copied_arguments;

  int a = 1;
  int b = 2;
  ASSERT_EQUAL(my_plus_proxy_ref{&a}(a, b), 3);
  static_assert(!allows_copied_arguments<my_plus_proxy_ref, int, int>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityProxyReferences);
