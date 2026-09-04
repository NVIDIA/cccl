// dlinfo / RTLD_DI_LINKMAP are behind _GNU_SOURCE, which selects declarations in
// glibc's features.h and so has to be defined before the first libc header in the
// file, not next to <dlfcn.h> below. Both clang and gcc predefine it for C++, so
// this only matters for a compiler that does not.
#ifndef _WIN32
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif
#endif

#include <cstddef>
#include <cstdio>
#include <string>

#include <hostjit/loader.hpp>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <link.h>
#endif

namespace hostjit
{
namespace
{
void* rawSymbol(void* handle, const char* name)
{
#ifdef _WIN32
  return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

void closeHandle(void* handle)
{
#ifdef _WIN32
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

#ifdef _WIN32
std::string getWindowsError()
{
  DWORD error = GetLastError();
  if (error == 0)
  {
    return "";
  }

  LPSTR buffer = nullptr;
  DWORD size   = FormatMessageA(
    FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
    nullptr,
    error,
    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
    reinterpret_cast<LPSTR>(&buffer),
    0,
    nullptr);

  std::string message;
  if (size > 0 && buffer)
  {
    message = std::string(buffer, size);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
    {
      message.pop_back();
    }
    LocalFree(buffer);
  }
  else
  {
    message = "Unknown error (code: " + std::to_string(error) + ")";
  }

  return message;
}
#endif
} // anonymous namespace

DynamicLibrary::DynamicLibrary()
    : handle_(nullptr)
    , module_fini_(nullptr)
    , module_sync_(nullptr)
    , unregistered_(false)
{}

DynamicLibrary::~DynamicLibrary()
{
  // A failed unload has nowhere to report to from here and no one left to retry, so
  // the image stays mapped -- with its fatbin still registered if it was the first
  // wait that refused, without it if it was the second. unload() has already said
  // which on stderr. Callers that need to know call unload() themselves.
  unload();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_)
    , module_fini_(other.module_fini_)
    , module_sync_(other.module_sync_)
    , unregistered_(other.unregistered_)
    , last_error_(std::move(other.last_error_))
{
  other.handle_       = nullptr;
  other.module_fini_  = nullptr;
  other.module_sync_  = nullptr;
  other.unregistered_ = false;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept
{
  if (this != &other)
  {
    if (!unload())
    {
      // A module that would not tear down cannot be overwritten here: this is the
      // only handle to an image that is still mapped, and with it gone nothing could
      // ever retry. So the two states change places instead of one replacing the
      // other. The assignment still goes ahead -- this object holds the module being
      // moved in -- and the module that refused stays owned by the object being moved
      // from, whose destructor tries once more. unload() has said on stderr why.
      std::swap(handle_, other.handle_);
      std::swap(module_fini_, other.module_fini_);
      std::swap(module_sync_, other.module_sync_);
      std::swap(unregistered_, other.unregistered_);
      last_error_.swap(other.last_error_);
      return *this;
    }
    handle_             = other.handle_;
    module_fini_        = other.module_fini_;
    module_sync_        = other.module_sync_;
    unregistered_       = other.unregistered_;
    last_error_         = std::move(other.last_error_);
    other.handle_       = nullptr;
    other.module_fini_  = nullptr;
    other.module_sync_  = nullptr;
    other.unregistered_ = false;
  }
  return *this;
}

bool DynamicLibrary::load(const std::string& library_path)
{
  if (!unload())
  {
    return false;
  }

#ifdef _WIN32
  SetLastError(0);
  handle_ = static_cast<void*>(LoadLibraryA(library_path.c_str()));

  if (!handle_)
  {
    last_error_ = getWindowsError();
    if (last_error_.empty())
    {
      last_error_ = "Unknown LoadLibrary error";
    }
    return false;
  }
  // Nothing to do after LoadLibrary: the DLL names its own entry point, which runs
  // the static constructors, so the fatbin is already registered. Running them from
  // here as well would register it twice.
#else
  dlerror();
  handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (!handle_)
  {
    const char* error = dlerror();
    last_error_       = error ? error : "Unknown dlopen error";
    return false;
  }
#endif

  // Every library the compiler produces exports these two, and unload() drives the module's
  // CUDA teardown through them. Without them the module cannot be torn down at all, so the
  // image is refused here, where the caller gets an error, rather than at teardown where there
  // is nothing left to report to.
  //
  // It is left mapped on purpose: loading it has run its constructors, so a library from an
  // older compiler has registered its fatbin by now, and closing it would strand that
  // registration, with no teardown entry point left to undo it first. Dropping the handle
  // costs the image's memory for the life of the process, which is the recoverable side.
  module_fini_ = reinterpret_cast<void (*)()>(rawSymbol(handle_, "__cudacc_module_fini"));
  module_sync_ = reinterpret_cast<int (*)()>(rawSymbol(handle_, "__cudacc_module_sync"));
  if (!module_fini_ || !module_sync_)
  {
    const char* const missing = !module_fini_ ? "__cudacc_module_fini" : "__cudacc_module_sync";
    last_error_   = std::string("Not a JIT module produced by this compiler: ") + missing
                  + " is missing; the image is left mapped, since unmapping it could strand a fatbin registration";
    handle_       = nullptr;
    module_fini_  = nullptr;
    module_sync_  = nullptr;
    unregistered_ = false;
    return false;
  }

  last_error_.clear();
  return true;
}

void* DynamicLibrary::getSymbol(const std::string& symbol_name)
{
  if (!handle_)
  {
    last_error_ = "Library not loaded";
    return nullptr;
  }

#ifdef _WIN32
  SetLastError(0);
  void* symbol = reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), symbol_name.c_str()));

  if (!symbol)
  {
    last_error_ = getWindowsError();
    if (last_error_.empty())
    {
      last_error_ = "Symbol not found: " + symbol_name;
    }
    return nullptr;
  }
#else
  dlerror();
  void* symbol = dlsym(handle_, symbol_name.c_str());

  const char* error = dlerror();
  if (error)
  {
    last_error_ = error;
    return nullptr;
  }
#endif

  last_error_.clear();
  return symbol;
}

bool DynamicLibrary::isLoaded() const
{
  return handle_ != nullptr;
}

std::string DynamicLibrary::getLastError() const
{
  return last_error_;
}

bool DynamicLibrary::unload()
{
  if (!handle_)
  {
    last_error_.clear();
    return true;
  }

  // Kernel launches are asynchronous and the runtime holds a pointer into the module's
  // embedded fatbin, while dlclose / FreeLibrary unmaps the image at once, so the module's
  // GPU work has to finish before its memory goes away. The barrier is the module's own
  // __cudacc_module_sync rather than a direct call: this file then needs no cudart of its
  // own, and the call lands in the runtime the module registered its fatbin with, of which
  // a process can hold more than one, each with its own registry and pending-unload queue.
  //
  // A failed wait stops the unload. cudaDeviceSynchronize refuses outright while the calling
  // thread is capturing a stream, and any other failure leaves it equally unknown from here
  // whether work is still running. Keeping the image mapped costs the module's memory until
  // the caller retries; unmapping it under a running kernel is the crash this path exists to
  // prevent. Which of the two waits failed decides what is left behind, told apart below.
  //
  // cudaErrorCudartUnloading is not a failure: the runtime is being torn down because the
  // process is exiting, so no work of ours is left to wait for and no later CUDA call is
  // coming to drain the queued unload. Only the waits are skipped for it; the teardown below
  // still runs here, with the image mapped and this thread off the loader lock, rather than
  // from the OS hook during the unmap. The value is spelled out because this file links no
  // cudart of its own.
  constexpr int cudart_unloading = 4; // cudaErrorCudartUnloading
  bool runtime_unloading         = false;

  // Skipped by a retry after the second wait failed: the teardown has run by then,
  // and running it again would be a no-op anyway, since it is one-shot. What such a
  // retry owes is the wait below, which is the one that did not happen.
  if (!unregistered_)
  {
    const int wait_status = module_sync_();
    if (wait_status != 0 && wait_status != cudart_unloading)
    {
      last_error_ =
        "the wait before unload failed (" + std::to_string(wait_status) + "), the module stays loaded and registered";
      std::fprintf(stderr, "hostjit: %s\n", last_error_.c_str());
      return false;
    }
    runtime_unloading = wait_status == cudart_unloading;

    // The module unregisters its fatbin on its own when the OS unloads it, but we
    // want that done while it is still mapped, so that the flush below can give the
    // device state back at unload time. The module exports __cudacc_module_fini()
    // for exactly this; it is one-shot, so the OS hook afterwards finds nothing to do.
    module_fini_();
    unregistered_ = true;
  }

  if (!runtime_unloading)
  {
    // Unregistering only queues the module for unload; the runtime issues the driver call
    // from its next entry point, so one more call here, with the image still mapped, is what
    // gives the device state back at unload time. Nothing public says when the runtime
    // issues that call or how to force it, so this rests on measured behaviour.
    //
    // A failure here stops the unload the way the first one does, though what it costs is
    // different: the unmap itself is safe by now, the teardown having only unregistered, so
    // what a failure means is that the device state has not come back. The message says so --
    // the fatbin is gone, the module can no longer be launched from, and only the unmap is
    // outstanding, which is where a retry resumes.
    const int flush_status = module_sync_();
    if (flush_status != 0 && flush_status != cudart_unloading)
    {
      last_error_ = "the wait after unregistering the module failed (" + std::to_string(flush_status)
                  + "), the module is unregistered and stays mapped; unload it again to finish";
      std::fprintf(stderr, "hostjit: %s\n", last_error_.c_str());
      return false;
    }
  }

  // The fatbin is unregistered, so it is now safe to unmap the module.
  closeHandle(handle_);
  handle_       = nullptr;
  module_fini_  = nullptr;
  module_sync_  = nullptr;
  unregistered_ = false;
  last_error_.clear();
  return true;
}

std::string DynamicLibrary::getLoadedModulePath() const
{
  if (!handle_)
  {
    return {};
  }
#ifdef _WIN32
  // A path that does not fit is truncated silently, and the call then reports the
  // buffer size rather than failing, so grow the buffer until the result fits. The
  // cap is the longest path Windows accepts.
  constexpr std::size_t max_path_bytes = 32768;
  std::string path(MAX_PATH, '\0');
  for (;;)
  {
    const DWORD n = GetModuleFileNameA(static_cast<HMODULE>(handle_), path.data(), static_cast<DWORD>(path.size()));
    if (n == 0)
    {
      return {};
    }
    if (n < path.size())
    {
      path.resize(n);
      return path;
    }
    if (path.size() >= max_path_bytes)
    {
      return {};
    }
    path.resize(path.size() * 2);
  }
#else
  struct link_map* lm = nullptr;
  if (dlinfo(handle_, RTLD_DI_LINKMAP, &lm) == 0 && lm != nullptr && lm->l_name != nullptr)
  {
    return std::string(lm->l_name);
  }
  return {};
#endif
}
} // namespace hostjit
