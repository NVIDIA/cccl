// %PARAM% TEST_ERR err 0:1:2:3

// This compilation sometimes passes, sometimes fails.
// It's role is to ensure that exit code is not checked for regex matches and binary objects are cleaned
// before each test run.
// This allows the failure machinery to test for non-fatal warnings.
int main()
{
  // Used if not specified otherwise:
  // expected-error {{"fail generic"}}

#if TEST_ERR == 0
#  pragma message "fail zero" // expected-error-0 {{"fail zero"}}
#elif TEST_ERR == 1
#  pragma message "fail generic"
#elif TEST_ERR == 2
  static_assert(false, "fail two"); // expected-error-2 {{"fail two"}}
#elif TEST_ERR == 3
  static_assert(false, "fail generic");
#endif
}
