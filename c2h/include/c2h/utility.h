// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <typeinfo>
#ifdef __GNUC__
#  include <cxxabi.h>
#endif // __GNUC__

namespace c2h
{
// TODO(bgruber): duplicated version of thrust/testing/unittest/system.h
inline std::string demangle(const char* name)
{
#if __GNUC__ && !_NVHPC_CUDA
  int status     = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);
  return result;
#else // __GNUC__ && !_NVHPC_CUDA
  return name;
#endif // __GNUC__ && !_NVHPC_CUDA
}

// TODO(bgruber): duplicated version of thrust/testing/unittest/util.h
template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
}
} // namespace c2h
