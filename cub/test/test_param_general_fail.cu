// %PARAM% TEST_ERR err 0:1:2:3

int main()
{
  // Used if not specified otherwise:
  // expected-error {{"fail generic"}}

#if TEST_ERR == 0
  static_assert(false, "fail zero"); // expected-error-0 {{"fail zero"}}
#elif TEST_ERR == 1
  static_assert(false, "fail generic");
#elif TEST_ERR == 2
  static_assert(false, "fail two"); // expected-error-2 {{"fail two"}}
#elif TEST_ERR == 3
  static_assert(false, "fail generic");
#endif
}
