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
  UnitTestException& operator<<(const T& t) &
  {
    std::ostringstream oss;
    oss << t;
    message += oss.str();
    return *this;
  }

  // The rvalue overload returns by value so that `throw UnitTestException{} << ...` throws an
  // anonymous temporary instead of an lvalue reference.
  template <typename T>
  UnitTestException operator<<(const T& t) &&
  {
    *this << t;
    return std::move(*this);
  }
};

class UnitTestError : public UnitTestException
{
public:
  UnitTestError() = default;
  UnitTestError(const std::string& msg)
      : UnitTestException(msg)
  {}

  template <typename T>
  UnitTestError& operator<<(const T& t) &
  {
    UnitTestException::operator<<(t);
    return *this;
  }

  template <typename T>
  UnitTestError operator<<(const T& t) &&
  {
    UnitTestException::operator<<(t);
    return std::move(*this);
  }
};

class UnitTestFailure : public UnitTestException
{
public:
  UnitTestFailure() = default;
  UnitTestFailure(const std::string& msg)
      : UnitTestException(msg)
  {}

  template <typename T>
  UnitTestFailure& operator<<(const T& t) &
  {
    UnitTestException::operator<<(t);
    return *this;
  }

  template <typename T>
  UnitTestFailure operator<<(const T& t) &&
  {
    UnitTestException::operator<<(t);
    return std::move(*this);
  }
};

class UnitTestKnownFailure : public UnitTestException
{
public:
  UnitTestKnownFailure() = default;
  UnitTestKnownFailure(const std::string& msg)
      : UnitTestException(msg)
  {}

  template <typename T>
  UnitTestKnownFailure& operator<<(const T& t) &
  {
    UnitTestException::operator<<(t);
    return *this;
  }

  template <typename T>
  UnitTestKnownFailure operator<<(const T& t) &&
  {
    UnitTestException::operator<<(t);
    return std::move(*this);
  }
};
} // end namespace unittest
