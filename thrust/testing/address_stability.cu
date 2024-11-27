#include <cuda/functional>

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
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  // libcu++ function objects with known types
  static_assert(proclaims_copyable_arguments<::cuda::std::plus<int>>::value, "");
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<>>::value, "");

  // libcu++ function objects with unknown types
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<addable>>::value, "");
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<>>::value, "");

  // libcu++ function objects with unknown types and opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(cuda::std::plus<addable>{}))>::value,
                "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(cuda::std::plus<>{}))>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityLibcuxx);

void TestAddressStabilityThrust()
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  // thrust function objects with known types
  static_assert(proclaims_copyable_arguments<thrust::plus<int>>::value, "");
  static_assert(!proclaims_copyable_arguments<thrust::plus<>>::value, "");

  // thrust function objects with unknown types
  static_assert(!proclaims_copyable_arguments<thrust::plus<addable>>::value, "");
  static_assert(!proclaims_copyable_arguments<thrust::plus<>>::value, "");

  // thrust function objects with unknown types and opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<addable>{}))>::value,
                "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(thrust::plus<>{}))>::value, "");
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

struct pathological_plus
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
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  // by-value overload
  static_assert(proclaims_copyable_arguments<my_plus<int>>::value, "");

  // by-value overload with opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int>{}))>::value, "");

  // by-reference overload
  static_assert(!proclaims_copyable_arguments<my_plus<int&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<const int&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<int&&>>::value, "");
  static_assert(!proclaims_copyable_arguments<my_plus<const int&&>>::value, "");

  // by-reference overload with opt-in
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<const int&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<int&&>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus<const int&&>{}))>::value, "");

  // pathological overloaded set
  static_assert(!proclaims_copyable_arguments<pathological_plus>::value, "");

  // call operator is a template
  static_assert(!proclaims_copyable_arguments<my_generic_plus>::value, ""); // may be solvable if we know T
  static_assert(!proclaims_copyable_arguments<my_generic_plus_lref>::value, "");
  static_assert(!proclaims_copyable_arguments<my_generic_plus_rref>::value, "");
  static_assert(!proclaims_copyable_arguments<my_generic_plus_clref>::value, "");
  static_assert(!proclaims_copyable_arguments<my_generic_plus_crref>::value, "");
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
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  // user-defined function types
  static_assert(proclaims_copyable_arguments<int(int, int)>::value, "");
  static_assert(!proclaims_copyable_arguments<int(const int&, const int&)>::value, "");

  static_assert(proclaims_copyable_arguments<decltype(my_plus_func)>::value, "");
  static_assert(!proclaims_copyable_arguments<decltype(my_plus_func_ref)>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(my_plus_func_ref))>::value, "");

  // user-defined function pointer types
  static_assert(proclaims_copyable_arguments<int (*)(int, int)>::value, "");
  static_assert(!proclaims_copyable_arguments<int (*)(const int&, const int&)>::value, "");

  static_assert(proclaims_copyable_arguments<decltype(&my_plus_func)>::value, "");
  static_assert(!proclaims_copyable_arguments<decltype(&my_plus_func_ref)>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(&my_plus_func_ref))>::value, "");

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
  using ::cuda::proclaims_copyable_arguments;

  int a = 1;
  int b = 2;
  ASSERT_EQUAL(my_plus_proxy_ref{&a}(a, b), 3);
  static_assert(!proclaims_copyable_arguments<my_plus_proxy_ref>::value, "");
}
DECLARE_UNITTEST(TestAddressStabilityProxyReferences);
