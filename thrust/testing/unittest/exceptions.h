#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace unittest
{
class UnitTestException
{
public:
  std::string message;

  UnitTestException() {}
  UnitTestException(const std::string& msg)
      : message(msg)
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

#if _CCCL_HAS_INT128()
  UnitTestException& operator<<(const __int128_t& t)
  {
    std::ostringstream oss;
    oss << static_cast<int64_t>(t);
    message += oss.str();
    return *this;
  }
  UnitTestException& operator<<(const __uint128_t& t)
  {
    std::ostringstream oss;
    oss << static_cast<uint64_t>(t);
    message += oss.str();
    return *this;
  }
#endif // _CCCL_HAS_INT128()
};

class UnitTestError : public UnitTestException
{
public:
  UnitTestError() {}
  UnitTestError(const std::string& msg)
      : UnitTestException(msg)
  {}
};

class UnitTestFailure : public UnitTestException
{
public:
  UnitTestFailure() {}
  UnitTestFailure(const std::string& msg)
      : UnitTestException(msg)
  {}
};

class UnitTestKnownFailure : public UnitTestException
{
public:
  UnitTestKnownFailure() {}
  UnitTestKnownFailure(const std::string& msg)
      : UnitTestException(msg)
  {}
};
} // end namespace unittest
