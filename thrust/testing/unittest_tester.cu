#include <unittest/unittest.h>

TEST_CASE("TestAssertEqual", "[unittest_tester]")
{
  REQUIRE(0 == 0);
  REQUIRE(1 == 1);
  REQUIRE(-15.0f == -15.0f);
}

TEST_CASE("TestAssertLEqual", "[unittest_tester]")
{
  REQUIRE(0 <= 1);
  REQUIRE(0 <= 0);
}

TEST_CASE("TestAssertGEqual", "[unittest_tester]")
{
  REQUIRE(1 >= 0);
  REQUIRE(0 >= 0);
}

TEST_CASE("TestAssertLess", "[unittest_tester]")
{
  REQUIRE(0 < 1);
}

TEST_CASE("TestAssertGreater", "[unittest_tester]")
{
  REQUIRE(1 > 0);
}

TEST_CASE("TestTypeName", "[unittest_tester]")
{
  REQUIRE(unittest::type_name<char>() == "char");
  REQUIRE(unittest::type_name<signed char>() == "signed char");
  REQUIRE(unittest::type_name<unsigned char>() == "unsigned char");
  REQUIRE(unittest::type_name<int>() == "int");
  REQUIRE(unittest::type_name<float>() == "float");
  REQUIRE(unittest::type_name<double>() == "double");
}
