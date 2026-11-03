#include <cuda/__functional/address_stability.h>

#include <unittest/unittest.h>

struct addable
{
  _CCCL_HOST_DEVICE friend auto operator+(const addable&, const addable&) -> addable
  {
    return addable{};
  }
};

TEST_CASE("TestAddressStabilityLibcuxx", "[address_stability]")
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments_v;

  // libcu++ function objects with known types
  static_assert(proclaims_copyable_arguments_v<::cuda::std::plus<int>>);
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<>>);

  // libcu++ function objects with unknown types
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<addable>>);
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<>>);

  // libcu++ function objects with unknown types and opt-in
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(cuda::std::plus<addable>{}))>);
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(cuda::std::plus<>{}))>);
}

TEST_CASE("TestAddressStabilityThrust", "[address_stability]")
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments_v;

  // thrust function objects with known types
  static_assert(proclaims_copyable_arguments_v<::cuda::std::plus<int>>);
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<>>);

  // thrust function objects with unknown types
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<addable>>);
  static_assert(!proclaims_copyable_arguments_v<::cuda::std::plus<>>);

  // thrust function objects with unknown types and opt-in
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(::cuda::std::plus<addable>{}))>);
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(::cuda::std::plus<>{}))>);
}

template <typename T>
struct my_plus
{
  _CCCL_HOST_DEVICE auto operator()(T a, T b) const -> T
  {
    return a + b;
  }
};

TEST_CASE("TestAddressStabilityUserDefinedFunctionObject", "[address_stability]")
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments_v;

  // by-value overload
  static_assert(!proclaims_copyable_arguments_v<my_plus<int>>);

  // by-value overload with opt-in
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(my_plus<int>{}))>);

  // by-reference overload
  static_assert(!proclaims_copyable_arguments_v<my_plus<int&>>);
  static_assert(!proclaims_copyable_arguments_v<my_plus<const int&>>);
  static_assert(!proclaims_copyable_arguments_v<my_plus<int&&>>);
  static_assert(!proclaims_copyable_arguments_v<my_plus<const int&&>>);

  // by-reference overload with opt-in
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(my_plus<int&>{}))>);
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(my_plus<const int&>{}))>);
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(my_plus<int&&>{}))>);
  static_assert(proclaims_copyable_arguments_v<decltype(proclaim_copyable_arguments(my_plus<const int&&>{}))>);
}

void test_address_stability_lambda()
{
  using ::cuda::proclaim_copyable_arguments;
  using ::cuda::proclaims_copyable_arguments_v;

  {
    auto l = [](const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments_v<decltype(l)>);
    auto pr_l = proclaim_copyable_arguments(l);
    REQUIRE(pr_l(3) == 5);
    static_assert(proclaims_copyable_arguments_v<decltype(pr_l)>);
  }

  {
    auto l = [] _CCCL_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments_v<decltype(l)>);
    [[maybe_unused]] auto pr_device_l = proclaim_copyable_arguments(l);
    static_assert(proclaims_copyable_arguments_v<decltype(pr_device_l)>);
  }

  {
    auto l = [] _CCCL_HOST_DEVICE(const int& i) {
      return i + 2;
    };
    static_assert(!proclaims_copyable_arguments_v<decltype(l)>);
    auto pr_l = proclaim_copyable_arguments(l);
    REQUIRE(pr_l(3) == 5);
    static_assert(proclaims_copyable_arguments_v<decltype(pr_l)>);
  }
}
TEST_CASE("test_address_stability_lambda", "[address_stability]")
{
  test_address_stability_lambda();
}
