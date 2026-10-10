#pragma once

#include <string>

namespace hostjit
{
class DynamicLibrary
{
public:
  DynamicLibrary();
  ~DynamicLibrary();

  // Disable copy
  DynamicLibrary(const DynamicLibrary&)            = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

  // Enable move
  DynamicLibrary(DynamicLibrary&& other) noexcept;
  DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

  // Load a shared library
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

  // Unload the library
  void unload();

private:
  void* handle_;
  std::string last_error_;
};
} // namespace hostjit
