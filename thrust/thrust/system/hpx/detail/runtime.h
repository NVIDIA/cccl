/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file runtime.h
 *  \brief Implementation of the HPX runtime startup/finalization.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <hpx/manage_runtime.hpp>

///////////////////////////////////////////////////////////////////////////////
// Store the command line arguments in global variables to make them available
// to the startup code.

#if defined(linux) || defined(__linux) || defined(__linux__)

inline int thrust_system_hpx__argc    = 0;
inline char** thrust_system_hpx__argv = nullptr;

inline void thrust_system_hpx__set_argc_argv(int argc, char* argv[], char*[])
{
  thrust_system_hpx__argc = argc;
  thrust_system_hpx__argv = argv;
}

__attribute__((section(".preinit_array"))) inline void (*thrust_system_hpx__set_global_argc_argv)(
  int, char*[], char*[]) = &thrust_system_hpx__set_argc_argv;

#elif defined(__APPLE__)

#  include <crt_externs.h>

inline int thrust_system_hpx__get_arraylen(char** argv)
{
  int count = 0;
  if (nullptr != argv)
  {
    while (nullptr != argv[count])
    {
      ++count;
    }
  }
  return count;
}

inline int thrust_system_hpx__argc    = thrust_system_hpx__get_arraylen(*_NSGetArgv());
inline char** thrust_system_hpx__argv = *_NSGetArgv();

#elif defined(_WIN32)

inline int thrust_system_hpx__argc    = __argc;
inline char** thrust_system_hpx__argv = __argv;

#endif

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

struct init_runtime
{
  ::hpx::manage_runtime runtime{};

  init_runtime()
  {
    ::hpx::init_params init_args;
    init_args.cfg = {
      // allow for unknown command line options
      "hpx.commandline.allow_unknown!=1",
      // disable HPX' short options
      "hpx.commandline.aliasing!=0",
    };
    init_args.mode = ::hpx::runtime_mode::default_;

    if (!runtime.start(thrust_system_hpx__argc, thrust_system_hpx__argv, init_args))
    {
      // something went wrong while initializing the runtime, bail out
      std::abort();
    }
  }

  ~init_runtime()
  {
    (void) runtime.stop();
  }

  static init_runtime& get()
  {
    // The HPX runtime implicitly depends on thread-local storage, making this object thread_local ensures the correct
    // sequencing of destructors. Since this function is only called from initialization of a global variable, only one
    // instance of the runtime will be created.
    static thread_local init_runtime m;
    return m;
  }
};

inline init_runtime& runtime = init_runtime::get();

template <typename F>
inline decltype(auto) run_as_hpx_thread(const F& f)
{
  if (::hpx::threads::get_self_ptr() != nullptr) // called from an HPX thread
  {
    return f();
  }

  return ::hpx::run_as_hpx_thread(f);
}

} // end namespace detail
} // end namespace hpx
} // end namespace system
THRUST_NAMESPACE_END
