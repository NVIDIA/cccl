// %PARAM% TEST_ERR err 0:1:2:3

#include "catch2_test_helper.h"

// This compilation sometimes passes, sometimes fails.
// It's role is to ensure that exit code is not checked for regex matches and binary objects are cleaned
// before each test run.
// This allows the failure machinery to test for non-fatal warnings.
TEST_CASE("FailModes", "[infra_fail]")
{
  // Used if not specified otherwise:
  // expected-error {{"fail generic"}}
#if TEST_ERR == 0 // exit code 0
#  pragma message("fail zero") // expected-error-0 {{"fail zero"}}
#elif TEST_ERR == 1 // exit code 0
#  pragma message("fail generic")
#elif TEST_ERR == 2 // exit code 1
  static_assert(false, "fail two"); // expected-error-2 {{"fail two"}}
#elif TEST_ERR == 3 // exit code 1
  static_assert(false, "fail generic");
#endif
}
