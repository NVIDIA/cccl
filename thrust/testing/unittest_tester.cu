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
  REQUIRE(0 <= 1);
  REQUIRE(0 <= 0);
}
DECLARE_UNITTEST(TestAssertLEqual);

void TestAssertGEqual()
{
  REQUIRE(1 >= 0);
  REQUIRE(0 >= 0);
}
DECLARE_UNITTEST(TestAssertGEqual);

void TestAssertLess()
{
  REQUIRE(0 < 1);
}
DECLARE_UNITTEST(TestAssertLess);

void TestAssertGreater()
{
  REQUIRE(1 > 0);
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
