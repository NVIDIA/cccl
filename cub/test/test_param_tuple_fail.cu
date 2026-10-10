// %PARAM% TEST_MODE mode 0:1
// %PARAM% TEST_VALUE,TEST_PAIRED_VALUE err 0=11,101:1=22,202

int main()
{
#if TEST_VALUE == 11 && TEST_PAIRED_VALUE == 101 && VAR_IDX == TEST_MODE * 2
  static_assert(false, "tuple zero"); // expected-error-0 {{"tuple zero"}}
#elif TEST_VALUE == 22 && TEST_PAIRED_VALUE == 202 && VAR_IDX == TEST_MODE * 2 + 1
  static_assert(false, "tuple one"); // expected-error-1 {{"tuple one"}}
#else
  static_assert(false, "invalid tuple parameterization");
#endif
}
