#pragma once

#include <string>

namespace hostjit
{
// Holds one JIT-compiled CUDA module. Not a general-purpose dynamic-library
// wrapper: unload() drives the module's CUDA teardown through entry points the
// produced library exports, so load() refuses an image that does not have them.
class DynamicLibrary
{
public:
  DynamicLibrary();
  ~DynamicLibrary();

  // Disable copy
  DynamicLibrary(const DynamicLibrary&)            = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  // Enable move. Assigning over a module whose unload is refused swaps the two
  // states rather than dropping the refused one: it is the only handle to a mapped
  // module, and the object moved from keeps it and retries when it is destroyed. So
  // a moved-from object is not necessarily empty -- see unload().
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  // Load a shared library. Fails on an image that does not export the module
  // teardown entry points, which is anything this compiler did not produce.
  bool load(const std::string& library_path);

  // Get a symbol (function or variable) by name
  void* getSymbol(const std::string& symbol_name);

  // Template helper to get function pointers with type safety
  template <typename FuncType>
  FuncType getFunction(const std::string& name)
  {
    return reinterpret_cast<FuncType>(getSymbol(name));
  }

  // Check if library is loaded
  bool isLoaded() const;

  // Get the last error message
  std::string getLastError() const;

  // Unload the library: wait for the module's GPU work, have the module unregister its
  // fatbin, force the runtime's pending unload out, then unmap. Returns false if a wait did
  // not happen; getLastError() says which one, and a later call resumes from there.
  //
  // What false leaves behind depends on the wait. The first comes before anything else, so
  // the module is untouched: mapped, registered, usable. The second comes after the fatbin
  // is unregistered, so the module is mapped but no longer something to launch from -- only
  // unloadable, and the only thing to do with the object is call unload() again.
  //
  // Nothing here holds off a concurrent user. A pointer taken from getSymbol() stays callable
  // until the unmap, and a call that starts after the wait is not covered by it, so the
  // caller has to have stopped using the module first -- the contract dlclose comes with.
  //
  // Call it from the thread and on the device where the module was used: the wait covers the
  // caller's current device only, so kernels this module still has running on another device
  // would be torn down under them. Waiting on every device would create a context on a
  // device that had none.
  bool unload();

  // Absolute path of the currently-loaded module as the OS sees it (via
  // GetModuleFileName on Windows / dlinfo on Linux), or empty if nothing is
  // loaded. Lets callers identify the module by its real name at runtime rather
  // than hard-coding it.
  std::string getLoadedModulePath() const;

private:
  void* handle_;
  // Resolved on load. module_fini_ asks the module to tear itself down (which
  // unregisters its fatbin) earlier than the OS would on unmap; module_sync_ is the
  // module's own call into the runtime it is bound to. See unload().
  void (*module_fini_)();
  int (*module_sync_)();
  // Set once the module has unregistered its fatbin, so that an unload resuming
  // after a refused second wait does not start the sequence over.
  bool unregistered_;
  std::string last_error_;
};
} // namespace hostjit
