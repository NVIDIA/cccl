#pragma once

#ifdef __GNUC__
#  include <cxxabi.h>
#endif // __GNUC__

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/type_traits>

#include <cstdlib>
#include <string>
#include <typeinfo>

namespace unittest
{
inline std::string demangle(const char* name)
{
  // for demangling the result of type_info.name() with msvc, type_info.name() is already demangled
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

template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
} // end type_name()

// Use this with counting_iterator to avoid generating a range larger than we can represent.
// TODO: This probably won't work for `half`.
template <typename T>
T truncate_to_max_representable(std::size_t n)
{
  if constexpr (::cuda::std::is_floating_point_v<T>)
  {
    return ::cuda::std::min<T>(static_cast<T>(n), ::cuda::std::numeric_limits<T>::max());
  }
  else
  {
    return static_cast<T>(
      ::cuda::std::min<std::size_t>(n, static_cast<std::size_t>(::cuda::std::numeric_limits<T>::max())));
  }
}

} // namespace unittest
