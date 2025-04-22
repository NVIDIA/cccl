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
  static_assert(proclaims_copyable_arguments<::cuda::std::plus<int>>::value, "");
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<>>::value, "");

  // thrust function objects with unknown types
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<addable>>::value, "");
  static_assert(!proclaims_copyable_arguments<::cuda::std::plus<>>::value, "");

  // thrust function objects with unknown types and opt-in
  static_assert(
    proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(::cuda::std::plus<addable>{}))>::value, "");
  static_assert(proclaims_copyable_arguments<decltype(proclaim_copyable_arguments(::cuda::std::plus<>{}))>::value, "");
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

void TestAddressStabilityUserDefinedFunctionObject()
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  // by-value overload
  static_assert(!proclaims_copyable_arguments<my_plus<int>>::value, "");

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
}
DECLARE_UNITTEST(TestAddressStabilityUserDefinedFunctionObject);

void TestAddressStabilityLambda()
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments;

  {
    auto l = [](const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    auto pr_l = proclaim_copyable_arguments(l);
    ASSERT_EQUAL(pr_l(3), 5);
    static_assert(proclaims_copyable_arguments<decltype(pr_l)>::value, "");
  }

  {
    auto l = [] _CCCL_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    [[maybe_unused]] auto pr_device_l = proclaim_copyable_arguments(l);
    static_assert(proclaims_copyable_arguments<decltype(pr_device_l)>::value, "");
  }

  {
    auto l = [] _CCCL_HOST_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments<decltype(l)>::value, "");
    auto pr_l = proclaim_copyable_arguments(l);
    ASSERT_EQUAL(pr_l(3), 5);
    static_assert(proclaims_copyable_arguments<decltype(pr_l)>::value, "");
  }
}
DECLARE_UNITTEST(TestAddressStabilityLambda);
