#pragma once

// for demangling the result of type_info.name()
// with msvc, type_info.name() is already demangled
#ifdef __GNUC__
#  include <cxxabi.h>
#endif // __GNUC__

#include <cstdlib>
#include <string>

namespace unittest
{
inline std::string demangle(const char* name)
{
#if __GNUC__ && !_NVHPC_CUDA
  int status     = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);
  return result;
#else
  return name;
#endif
}
} // namespace unittest
