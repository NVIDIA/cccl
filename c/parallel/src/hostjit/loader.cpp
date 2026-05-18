#include <hostjit/loader.hpp>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace hostjit
{
#ifdef _WIN32
namespace
{
// Run C++ static constructors in a DLL loaded with /NOENTRY /NODEFAULTLIB.
// The compiler places CUDA fatbin registration in the .CRT$XCU section.
// Without CRT startup, these never run, so we walk the merged .CRT section
// in the PE and call each non-null function pointer.
void runStaticInitializers(HMODULE module)
{
  auto base = reinterpret_cast<const unsigned char*>(module);
  auto dos  = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  auto nt   = reinterpret_cast<const IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
  auto sec  = IMAGE_FIRST_SECTION(nt);

  for (WORD i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++sec)
  {
    if (memcmp(sec->Name, ".CRT", 4) == 0)
    {
      using InitFunc = void(__cdecl*)();
      auto funcs     = reinterpret_cast<InitFunc*>(const_cast<unsigned char*>(base) + sec->VirtualAddress);
      size_t count   = sec->SizeOfRawData / sizeof(InitFunc);
      for (size_t j = 0; j < count; ++j)
      {
        if (funcs[j])
        {
          funcs[j]();
        }
      }
    }
  }
}

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
} // anonymous namespace
#endif

DynamicLibrary::DynamicLibrary()
    : handle_(nullptr)
{}

DynamicLibrary::~DynamicLibrary()
{
  unload();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_)
    , last_error_(std::move(other.last_error_))
{
  other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept
{
  if (this != &other)
  {
    unload();
    handle_       = other.handle_;
    last_error_   = std::move(other.last_error_);
    other.handle_ = nullptr;
  }
  return *this;
}

bool DynamicLibrary::load(const std::string& library_path)
{
  unload();

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

  // The DLL is linked with /NOENTRY (no CRT startup), so C++ static
  // constructors (e.g. CUDA fatbin registration) haven't run yet.
  runStaticInitializers(static_cast<HMODULE>(handle_));
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

void DynamicLibrary::unload()
{
  if (handle_)
  {
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
    handle_ = nullptr;
  }
  last_error_.clear();
}
} // namespace hostjit
