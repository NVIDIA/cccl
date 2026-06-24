#pragma once

#include <string>
#include <vector>

#include <libnvcc/libnvcc.h>

namespace hostjit::detail
{
struct LibnvccProgramGuard
{
  libnvccProgram program = nullptr;

  ~LibnvccProgramGuard()
  {
    libnvccDestroyProgram(&program);
  }
};

inline std::vector<const char*> make_libnvcc_option_ptrs(const std::vector<std::string>& options)
{
  std::vector<const char*> ptrs;
  ptrs.reserve(options.size());
  for (const auto& option : options)
  {
    ptrs.push_back(option.c_str());
  }
  return ptrs;
}

inline std::string get_libnvcc_program_log(libnvccProgram program)
{
  size_t log_size = 0;
  if (libnvccGetProgramLogSize(program, &log_size) != LIBNVCC_SUCCESS || log_size == 0)
  {
    return {};
  }
  std::string log(log_size, '\0');
  if (libnvccGetProgramLog(program, log.data()) != LIBNVCC_SUCCESS)
  {
    return {};
  }
  if (!log.empty() && log.back() == '\0')
  {
    log.pop_back();
  }
  return log;
}
} // namespace hostjit::detail
