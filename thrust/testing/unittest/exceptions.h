#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace unittest
{
class UnitTestException
{
public:
  std::string message;

  UnitTestException() = default;
  UnitTestException(std::string msg)
      : message(std::move(msg))
  {}

  friend std::ostream& operator<<(std::ostream& os, const UnitTestException& e)
  {
    return os << e.message;
  }

  template <typename T>
  UnitTestException& operator<<(const T& t)
  {
    std::ostringstream oss;
    oss << t;
    message += oss.str();
    return *this;
  }
};

class UnitTestError : public UnitTestException
{
public:
  UnitTestError() = default;
  UnitTestError(const std::string& msg)
      : UnitTestException(msg)
  {}
};

class UnitTestFailure : public UnitTestException
{
public:
  UnitTestFailure() = default;
  UnitTestFailure(const std::string& msg)
      : UnitTestException(msg)
  {}
};

class UnitTestKnownFailure : public UnitTestException
{
public:
  UnitTestKnownFailure() = default;
  UnitTestKnownFailure(const std::string& msg)
      : UnitTestException(msg)
  {}
};
} // end namespace unittest
