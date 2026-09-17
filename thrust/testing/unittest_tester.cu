#include <unittest/unittest.h>

void TestAssertEqual()
{
  REQUIRE(0 == 0);
  REQUIRE(1 == 1);
  REQUIRE(-15.0f == -15.0f);
}
DECLARE_UNITTEST(TestAssertEqual);

void TestAssertLEqual()
{
  ASSERT_LEQUAL(0, 1);
  ASSERT_LEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertLEqual);

void TestAssertGEqual()
{
  ASSERT_GEQUAL(1, 0);
  ASSERT_GEQUAL(0, 0);
}
DECLARE_UNITTEST(TestAssertGEqual);

void TestAssertLess()
{
  ASSERT_LESS(0, 1);
}
DECLARE_UNITTEST(TestAssertLess);

void TestAssertGreater()
{
  ASSERT_GREATER(1, 0);
}
DECLARE_UNITTEST(TestAssertGreater);

void TestTypeName()
{
  REQUIRE(unittest::type_name<char>() == "char");
  REQUIRE(unittest::type_name<signed char>() == "signed char");
  REQUIRE(unittest::type_name<unsigned char>() == "unsigned char");
  REQUIRE(unittest::type_name<int>() == "int");
  REQUIRE(unittest::type_name<float>() == "float");
  REQUIRE(unittest::type_name<double>() == "double");
}
DECLARE_UNITTEST(TestTypeName);
