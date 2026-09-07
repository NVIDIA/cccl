#pragma once

#include <cassert>
#include <string>
#include <vector>

#include <libnvcc/libnvcc.h>

namespace hostjit::detail
{
struct LibnvccProgramGuard
{
  libnvccProgram program = nullptr;

  LibnvccProgramGuard()                                      = default;
  LibnvccProgramGuard(const LibnvccProgramGuard&)            = delete;
  LibnvccProgramGuard& operator=(const LibnvccProgramGuard&) = delete;

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
  if (libnvccGetProgramLogSize(program, &log_size) != LIBNVCC_SUCCESS)
  {
    return {};
  }

  assert(log_size > 0 && "Log size should include NUL terminator");
  if (log_size == 1)
  {
    return {};
  }

  std::string log(log_size, '\0');
  [[maybe_unused]] auto res = libnvccGetProgramLog(program, log.data());
  assert(res == LIBNVCC_SUCCESS && "Copying the log failed even though size calculation succeeded?");
  assert(log.back() == '\0' && "libnvccGetProgramLog() should append a NUL character");
  log.pop_back(); // Drop the extra NUL.
  return log;
}
} // namespace hostjit::detail
