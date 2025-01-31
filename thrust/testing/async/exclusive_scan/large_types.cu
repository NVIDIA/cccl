#include <cuda/__cccl_config>

_CCCL_SUPPRESS_DEPRECATED_PUSH

#include <thrust/detail/config.h>

#if _CCCL_STD_VER >= 2014

#  include <async/exclusive_scan/mixin.h>
#  include <async/test_policy_overloads.h>
#  include <unittest/special_types.h>

// This test is an adaptation of TestScanWithLargeTypes from scan.cu.

// Need special initialization for the FixedVector type:
template <typename value_type>
struct device_vector_fill
{
  using input_type = thrust::device_vector<value_type>;

  static input_type generate_input(std::size_t num_values)
  {
    input_type input(num_values);
    thrust::fill(input.begin(), input.end(), value_type{2});
    return input;
  }
};

template <typename value_type, typename alternate_binary_op = thrust::maximum<>>
struct invoker
    : device_vector_fill<value_type>
    , testing::async::mixin::output::device_vector<value_type>
    , testing::async::exclusive_scan::mixin::postfix_args::all_overloads<value_type, alternate_binary_op>
    , testing::async::exclusive_scan::mixin::invoke_reference::host_synchronous<value_type>
    , testing::async::exclusive_scan::mixin::invoke_async::simple
    , testing::async::mixin::compare_outputs::assert_almost_equal_if_fp_quiet
{
  static std::string description()
  {
    return "scan with large value types.";
  }
};

struct test_large_types
{
  void operator()(std::size_t num_values) const
  {
    using testing::async::test_policy_overloads;

    test_policy_overloads<invoker<FixedVector<int, 1>>>::run(num_values);
    test_policy_overloads<invoker<FixedVector<int, 8>>>::run(num_values);
    test_policy_overloads<invoker<FixedVector<int, 32>>>::run(num_values);
    test_policy_overloads<invoker<FixedVector<int, 64>>>::run(num_values);
  }
};
DECLARE_UNITTEST(test_large_types);

#endif // C++14

// we need to leak the suppression on clang to suppresses warnings from the cudafe1.stub.c file
#if !_CCCL_COMPILER(CLANG)
_CCCL_SUPPRESS_DEPRECATED_POP
#endif
