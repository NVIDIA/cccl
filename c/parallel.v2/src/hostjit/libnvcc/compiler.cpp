#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <libnvcc/libnvcc.h>
#include <lld/Common/Driver.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/VirtualFileSystem.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>

// Selective target initialization (X86 for host, NVPTX for device)
extern "C" {
void LLVMInitializeX86TargetInfo();
void LLVMInitializeX86Target();
void LLVMInitializeX86TargetMC();
void LLVMInitializeX86AsmPrinter();
void LLVMInitializeX86AsmParser();
void LLVMInitializeNVPTXTargetInfo();
void LLVMInitializeNVPTXTarget();
void LLVMInitializeNVPTXTargetMC();
void LLVMInitializeNVPTXAsmPrinter();
}

#ifdef _WIN32
LLD_HAS_DRIVER(coff)
#else
LLD_HAS_DRIVER(elf)
#endif

#ifdef _WIN32
#  include <llvm/Object/COFFImportFile.h>
#endif

#include <charconv>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <nvFatbin.h>
#include <nvJitLink.h>

namespace libnvcc
{
static bool llvm_initialized = false;

static void initialize_llvm()
{
  if (llvm_initialized)
  {
    return;
  }

  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();

  llvm_initialized = true;
}

struct CompilerOptions
{
  std::string cuda_toolkit_path;
  std::string hostjit_include_path;
  std::string clang_headers_path;
  std::string device_pch_path;
  std::string host_pch_path;
  std::string entry_point_name;
  std::vector<std::string> system_include_paths;
  std::vector<std::string> include_paths;
  std::vector<std::string> library_paths;
  std::vector<std::string> device_bitcode_files;
  std::vector<std::string> device_ltoir_files;
  std::unordered_map<std::string, std::string> macro_definitions;
  std::vector<std::string> extra_clang_args;
  int sm_version         = 75;
  int optimization_level = 2;
  bool debug             = false;
  bool verbose           = false;
  bool trace_includes    = false;
  bool keep_artifacts    = false;
};

struct CompilationResult
{
  bool success = false;
  std::string object_file_path;
  std::string diagnostics;
};

struct BitcodeResult
{
  bool success = false;
  std::string diagnostics;
};

struct LinkResult
{
  bool success = false;
  std::string library_path;
  std::string diagnostics;
};

static bool pathExists(const std::filesystem::path& path);

static void addDefaultCudaLibraryPath(CompilerOptions& options)
{
  if (!options.cuda_toolkit_path.empty())
  {
    std::filesystem::path lib64_path = std::filesystem::path(options.cuda_toolkit_path) / "lib64";
    std::filesystem::path lib_path   = std::filesystem::path(options.cuda_toolkit_path) / "lib";

    if (pathExists(lib64_path))
    {
      options.library_paths.push_back(lib64_path.string());
    }
    else if (pathExists(lib_path))
    {
      options.library_paths.push_back(lib_path.string());
    }
  }
}

static void setDefaultOptions(CompilerOptions& options)
{
  if (const char* env = std::getenv("CUDA_PATH"))
  {
    options.cuda_toolkit_path = env;
  }
  else if (const char* env = std::getenv("CUDA_HOME"))
  {
    options.cuda_toolkit_path = env;
  }
#ifdef CUDA_TOOLKIT_PATH
  else
  {
    options.cuda_toolkit_path = CUDA_TOOLKIT_PATH;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_INCLUDE_PATH"))
  {
    options.hostjit_include_path = env;
  }
#ifdef HOSTJIT_INCLUDE_DIR
  else
  {
    options.hostjit_include_path = HOSTJIT_INCLUDE_DIR;
  }
#endif

  if (const char* env = std::getenv("HOSTJIT_CLANG_PATH"))
  {
    options.clang_headers_path = env;
  }
#ifdef CLANG_HEADERS_DIR
  else
  {
    options.clang_headers_path = CLANG_HEADERS_DIR;
  }
#endif
}

static bool pathExists(const std::filesystem::path& path)
{
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}

static std::filesystem::path tempDirectoryPath()
{
  std::error_code ec;
  auto path = std::filesystem::temp_directory_path(ec);
  if (!ec)
  {
    return path;
  }
#ifdef _WIN32
  if (const char* env = std::getenv("TEMP"))
  {
    return env;
  }
  if (const char* env = std::getenv("TMP"))
  {
    return env;
  }
#endif
  if (const char* env = std::getenv("TMPDIR"))
  {
    return env;
  }
  return ".";
}

static bool createDirectories(const std::filesystem::path& path, std::string& diagnostics)
{
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  if (ec)
  {
    diagnostics += "Failed to create directory " + path.string() + ": " + ec.message() + "\n";
    return false;
  }
  return true;
}

static void removeAll(const std::filesystem::path& path)
{
  std::error_code ec;
  std::filesystem::remove_all(path, ec);
}

template <typename Fn>
static void forEachDirectoryEntry(const std::filesystem::path& dir, Fn&& fn)
{
  std::error_code ec;
  for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec))
  {
    fn(*it);
  }
}

static bool parseInt(const std::string& value, int& out)
{
  if (value.empty())
  {
    return false;
  }

  int parsed        = 0;
  const char* begin = value.data();
  const char* end   = begin + value.size();
  auto [ptr, ec]    = std::from_chars(begin, end, parsed);
  if (ec != std::errc{} || ptr != end)
  {
    return false;
  }
  out = parsed;
  return true;
}

static bool parseGpuArchitecture(const std::string& value, int& sm)
{
  std::string arch = value;
  if (arch.starts_with("sm_"))
  {
    arch.erase(0, 3);
  }
  return parseInt(arch, sm);
}

static bool parseMacroDefinition(const std::string& value, CompilerOptions& options)
{
  if (value.empty())
  {
    return false;
  }
  auto eq = value.find('=');
  if (eq == std::string::npos)
  {
    options.macro_definitions[value] = "";
  }
  else if (eq == 0)
  {
    return false;
  }
  else
  {
    options.macro_definitions[value.substr(0, eq)] = value.substr(eq + 1);
  }
  return true;
}

static bool parseOptions(int num_options, const char* const* raw_options, CompilerOptions& options, std::string& error)
{
  if (num_options < 0)
  {
    error = "Option count must be non-negative";
    return false;
  }
  if (num_options > 0 && raw_options == nullptr)
  {
    error = "Options array is null";
    return false;
  }

  setDefaultOptions(options);

  auto value_after_equals = [](std::string_view option, std::string_view prefix) -> std::string {
    return std::string(option.substr(prefix.size()));
  };

  for (int i = 0; i < num_options; ++i)
  {
    if (raw_options[i] == nullptr)
    {
      error = "Option string is null";
      return false;
    }

    std::string_view option(raw_options[i]);
    if (option.starts_with("--cuda-path="))
    {
      options.cuda_toolkit_path = value_after_equals(option, "--cuda-path=");
    }
    else if (option.starts_with("--hostjit-include-path="))
    {
      options.hostjit_include_path = value_after_equals(option, "--hostjit-include-path=");
    }
    else if (option.starts_with("--clang-headers-path="))
    {
      options.clang_headers_path = value_after_equals(option, "--clang-headers-path=");
    }
    else if (option.starts_with("--system-include-path="))
    {
      options.system_include_paths.push_back(value_after_equals(option, "--system-include-path="));
    }
    else if (option.starts_with("-isystem") && option.size() > 8)
    {
      options.system_include_paths.emplace_back(option.substr(8));
    }
    else if (option == "-isystem")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-isystem requires an argument";
        return false;
      }
      options.system_include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--include-path="))
    {
      options.include_paths.push_back(value_after_equals(option, "--include-path="));
    }
    else if (option.starts_with("-I") && option.size() > 2)
    {
      options.include_paths.emplace_back(option.substr(2));
    }
    else if (option == "-I")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-I requires an argument";
        return false;
      }
      options.include_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--library-path="))
    {
      options.library_paths.push_back(value_after_equals(option, "--library-path="));
    }
    else if (option.starts_with("-L") && option.size() > 2)
    {
      options.library_paths.emplace_back(option.substr(2));
    }
    else if (option == "-L")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-L requires an argument";
        return false;
      }
      options.library_paths.emplace_back(raw_options[i]);
    }
    else if (option.starts_with("--device-bitcode="))
    {
      options.device_bitcode_files.push_back(value_after_equals(option, "--device-bitcode="));
    }
    else if (option.starts_with("--device-ltoir="))
    {
      options.device_ltoir_files.push_back(value_after_equals(option, "--device-ltoir="));
    }
    else if (option.starts_with("--define-macro="))
    {
      if (!parseMacroDefinition(value_after_equals(option, "--define-macro="), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-D") && option.size() > 2)
    {
      if (!parseMacroDefinition(std::string(option.substr(2)), options))
      {
        error = "Invalid macro definition: " + std::string(option);
        return false;
      }
    }
    else if (option == "-D")
    {
      if (++i >= num_options || raw_options[i] == nullptr || !parseMacroDefinition(raw_options[i], options))
      {
        error = "-D requires a macro definition";
        return false;
      }
    }
    else if (option.starts_with("--gpu-architecture="))
    {
      if (!parseGpuArchitecture(value_after_equals(option, "--gpu-architecture="), options.sm_version))
      {
        error = "Invalid GPU architecture: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("--optimization-level="))
    {
      if (!parseInt(value_after_equals(option, "--optimization-level="), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option.starts_with("-O") && option.size() > 2)
    {
      if (!parseInt(std::string(option.substr(2)), options.optimization_level))
      {
        error = "Invalid optimization level: " + std::string(option);
        return false;
      }
    }
    else if (option == "--debug")
    {
      options.debug = true;
    }
    else if (option == "--verbose")
    {
      options.verbose = true;
    }
    else if (option == "--trace-includes")
    {
      options.trace_includes = true;
    }
    else if (option == "--keep-artifacts")
    {
      options.keep_artifacts = true;
    }
    else if (option.starts_with("--entry-point="))
    {
      options.entry_point_name = value_after_equals(option, "--entry-point=");
    }
    else if (option.starts_with("--device-pch="))
    {
      options.device_pch_path = value_after_equals(option, "--device-pch=");
    }
    else if (option.starts_with("--host-pch="))
    {
      options.host_pch_path = value_after_equals(option, "--host-pch=");
    }
    else if (option.starts_with("-XClang="))
    {
      options.extra_clang_args.emplace_back(option.substr(8));
    }
    else if (option == "-XClang")
    {
      if (++i >= num_options || raw_options[i] == nullptr)
      {
        error = "-XClang requires an argument";
        return false;
      }
      options.extra_clang_args.emplace_back(raw_options[i]);
    }
    else
    {
      error = "Unknown option: " + std::string(option);
      return false;
    }
  }

  if (options.library_paths.empty())
  {
    addDefaultCudaLibraryPath(options);
  }

  return true;
}

static bool validateOptions(const CompilerOptions& options, std::string* error_message)
{
  if (options.cuda_toolkit_path.empty())
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path not found. Please pass --cuda-path or set CUDA_PATH/CUDA_HOME.";
    }
    return false;
  }

  if (!pathExists(options.cuda_toolkit_path))
  {
    if (error_message)
    {
      *error_message = "CUDA toolkit path does not exist: " + options.cuda_toolkit_path;
    }
    return false;
  }

  std::filesystem::path cuda_h = std::filesystem::path(options.cuda_toolkit_path) / "include" / "cuda.h";
  if (!pathExists(cuda_h))
  {
    if (error_message)
    {
      *error_message = "CUDA headers not found at: " + cuda_h.string();
    }
    return false;
  }

  for (const auto& include_path : options.include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "Include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& include_path : options.system_include_paths)
  {
    if (!pathExists(include_path))
    {
      if (error_message)
      {
        *error_message = "System include path does not exist: " + include_path;
      }
      return false;
    }
  }

  for (const auto& library_path : options.library_paths)
  {
    if (!pathExists(library_path))
    {
      if (error_message)
      {
        *error_message = "Library path does not exist: " + library_path;
      }
      return false;
    }
  }

  for (const auto& bitcode_path : options.device_bitcode_files)
  {
    if (!pathExists(bitcode_path))
    {
      if (error_message)
      {
        *error_message = "Device bitcode path does not exist: " + bitcode_path;
      }
      return false;
    }
  }

  for (const auto& ltoir_path : options.device_ltoir_files)
  {
    if (!pathExists(ltoir_path))
    {
      if (error_message)
      {
        *error_message = "Device LTOIR path does not exist: " + ltoir_path;
      }
      return false;
    }
  }

  if (!options.device_pch_path.empty() && !pathExists(options.device_pch_path))
  {
    if (error_message)
    {
      *error_message = "Device PCH path does not exist: " + options.device_pch_path;
    }
    return false;
  }

  if (!options.host_pch_path.empty() && !pathExists(options.host_pch_path))
  {
    if (error_message)
    {
      *error_message = "Host PCH path does not exist: " + options.host_pch_path;
    }
    return false;
  }

  if (options.sm_version < 30 || options.sm_version > 150)
  {
    if (error_message)
    {
      *error_message = "Invalid SM version: " + std::to_string(options.sm_version) + " (must be between 30 and 150)";
    }
    return false;
  }

  if (options.optimization_level < 0 || options.optimization_level > 3)
  {
    if (error_message)
    {
      *error_message =
        "Invalid optimization level: " + std::to_string(options.optimization_level) + " (must be between 0 and 3)";
    }
    return false;
  }

  return true;
}

static void appendExtraClangArgs(std::vector<std::string>& args, const CompilerOptions& options)
{
  args.insert(args.end(), options.extra_clang_args.begin(), options.extra_clang_args.end());
}

static void appendSystemIncludePaths(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& include_path : options.system_include_paths)
  {
    args.push_back("-internal-isystem");
    args.push_back(include_path);
  }
}

static void appendIncludePaths(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& include_path : options.include_paths)
  {
    args.push_back("-I" + include_path);
  }
}

static void appendMacroDefinitions(std::vector<std::string>& args, const CompilerOptions& options)
{
  for (const auto& [macro_name, macro_value] : options.macro_definitions)
  {
    if (macro_value.empty())
    {
      args.push_back("-D" + macro_name);
    }
    else
    {
      args.push_back("-D" + macro_name + "=" + macro_value);
    }
  }
}

#ifdef _WIN32
// Generate a minimal COFF import library for a given DLL.
// This allows linking without requiring the Windows SDK or MSVC .lib files.
// Symbols can be "name" or "name=dllexport" for aliasing.
static bool generateImportLib(
  const std::string& dll_name,
  const std::vector<std::string>& symbols,
  const std::string& output_path,
  bool data_only = false)
{
  std::vector<llvm::object::COFFShortExport> exports;
  for (const auto& sym : symbols)
  {
    llvm::object::COFFShortExport exp;
    auto eq = sym.find('=');
    if (eq != std::string::npos)
    {
      // "atexit=_crt_atexit" means: linker sees "atexit", DLL exports "_crt_atexit"
      exp.Name       = sym.substr(0, eq); // symbol name the linker resolves
      exp.ImportName = sym.substr(eq + 1); // actual DLL export name
    }
    else
    {
      exp.Name = sym;
    }
    exp.Data = data_only;
    exports.push_back(exp);
  }
  auto err = llvm::object::writeImportLibrary(
    dll_name,
    output_path,
    exports,
    llvm::COFF::IMAGE_FILE_MACHINE_AMD64,
    /*MinGW=*/false);
  if (err)
  {
    llvm::consumeError(std::move(err));
    return false;
  }
  return true;
}

// Find the actual DLL filename for cudart (e.g. "cudart64_13.dll") by
// scanning the CUDA toolkit bin directory.
static std::string findCudartDllName(const std::string& cuda_toolkit_path)
{
  namespace fs = std::filesystem;
  for (const auto& subdir : {"bin/x64", "bin"})
  {
    fs::path dir = fs::path(cuda_toolkit_path) / subdir;
    if (!pathExists(dir))
    {
      continue;
    }
    std::string cudart_name;
    forEachDirectoryEntry(dir, [&](const std::filesystem::directory_entry& entry) {
      auto name = entry.path().filename().string();
      if (cudart_name.empty() && name.starts_with("cudart64_") && name.ends_with(".dll"))
      {
        cudart_name = name;
      }
    });
    if (!cudart_name.empty())
    {
      return cudart_name;
    }
  }
  return "cudart64_12.dll"; // fallback
}
#endif

class CompilerImpl
{
public:
  CompilerImpl() {}

  // Write preamble to a persistent file and generate a PCH from it.
  // arg_strings[0] will be replaced with the persistent preamble path.
  bool generatePCH(const std::string& pch_source,
                   const std::string& pch_source_path,
                   const std::string& pch_output_path,
                   std::vector<std::string> arg_strings,
                   std::string& diagnostics)
  {
    // Write preamble to the persistent source path
    {
      std::ofstream f(pch_source_path);
      if (!f)
      {
        diagnostics += "Failed to write PCH preamble to " + pch_source_path;
        return false;
      }
      f << pch_source;
    }

    // Replace the source file arg with the persistent path
    arg_strings[0] = pch_source_path;

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);
    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors = false;
    auto* diag_printer   = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output + "\nFailed to create PCH compiler invocation";
      return false;
    }

    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.createFileManager();
    compiler.getFrontendOpts().OutputFile = pch_output_path;

    clang::GeneratePCHAction pch_action;
    bool success = compiler.ExecuteAction(pch_action);

    diag_stream.flush();
    diagnostics += diag_output;

    return success;
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSWithSource(const std::string& source_code, const std::string& virtual_path)
  {
    auto mem_fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    mem_fs->addFile(virtual_path, 0, llvm::MemoryBuffer::getMemBuffer(source_code));

    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(mem_fs);
    return overlay;
  }

  bool compileDeviceToPTX(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& output_ptx,
    const CompilerOptions& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_ptx).parent_path().string();
    std::string source_file = temp_dir + "/" + input_file;

    std::string resource_dir = CLANG_RESOURCE_DIR;

    // PTX version floor is 7.8. Some generated device code uses features
    // added in PTX 7.6 (e.g. `bmsk`), so older versions can fail to assemble
    // even on sm_75/sm_80.
    int ptx_version = 78;
    if (config.sm_version >= 120)
    {
      ptx_version = 87;
    }
    else if (config.sm_version >= 100)
    {
      ptx_version = 85;
    }
    else if (config.sm_version >= 90)
    {
      ptx_version = 80;
    }

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-aux-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-S");
    arg_strings.push_back("-aux-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-is-device");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mlink-builtin-bitcode");
    arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("sm_" + std::to_string(config.sm_version));
    arg_strings.push_back("-target-feature");
    arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");

    std::vector<std::string> bitcode_files_to_link = config.device_bitcode_files;

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("--offload-new-driver");
    arg_strings.push_back("-fskip-odr-check-in-gmf");
    arg_strings.push_back("-fcxx-exceptions");
    arg_strings.push_back("-fexceptions");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::string device_pch_path = config.device_pch_path;

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    if (config.verbose)
    {
      diagnostics += "Device args: ";
      for (const auto& arg : arg_strings)
      {
        diagnostics += arg + " ";
      }
      diagnostics += "\n";
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output;
      diagnostics += "\nFailed to create device compiler invocation";
      return false;
    }

    // --- PCH: load cached device PCH ---
    if (!device_pch_path.empty() && pathExists(device_pch_path))
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = device_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();
    compiler.getFrontendOpts().OutputFile = output_ptx;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Device Header Search Paths ===\n";
      const auto& hso = invocation.getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

    llvm::LLVMContext llvm_context;

    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = compiler.ExecuteAction(emit_llvm_action);

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Device Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_llvm_action.takeModule();
      if (mod)
      {
        for (const auto& bc_file : bitcode_files_to_link)
        {
          llvm::SMDiagnostic err;
          auto bc_mod = llvm::parseIRFile(bc_file, err, llvm_context);
          if (bc_mod)
          {
            if (llvm::Linker::linkModules(*mod, std::move(bc_mod)))
            {
              diagnostics += "Failed to link bitcode: " + bc_file + "\n";
              success = false;
              break;
            }
          }
          else
          {
            std::string err_msg;
            llvm::raw_string_ostream err_stream(err_msg);
            err.print("hostjit", err_stream);
            diagnostics += "Failed to parse bitcode: " + bc_file + "\n" + err_msg + "\n";
            success = false;
            break;
          }
        }

        // Re-link libdevice to resolve any new references (e.g. __nv_pow)
        // introduced by the extra bitcode modules.
        if (success && !bitcode_files_to_link.empty())
        {
          std::string libdevice_path = config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc";
          llvm::SMDiagnostic err;
          auto libdevice = llvm::parseIRFile(libdevice_path, err, llvm_context);
          if (libdevice)
          {
            // Use AppendToUsed to avoid internalization issues
            llvm::Linker::linkModules(*mod, std::move(libdevice), llvm::Linker::LinkOnlyNeeded);
          }
        }

        if (success)
        {
          std::string err_str;
          const llvm::Target* target = llvm::TargetRegistry::lookupTarget(mod->getTargetTriple(), err_str);
          if (target)
          {
            llvm::TargetOptions opt;
            auto tm = target->createTargetMachine(
              mod->getTargetTriple(),
              "sm_" + std::to_string(config.sm_version),
              "+ptx" + std::to_string(ptx_version),
              opt,
              llvm::Reloc::PIC_);
            if (tm)
            {
              mod->setDataLayout(tm->createDataLayout());

              // Run optimization passes after linking to inline user-provided
              // operations (from bitcode or embedded C++ source).
              if (!config.entry_point_name.empty())
              {
                // Internalize all functions except the entry point and
                // GPU kernels, so the optimizer can inline the linked
                // bitcode functions.
                for (auto& F : *mod)
                {
                  if (!F.isDeclaration() && F.getLinkage() == llvm::GlobalValue::ExternalLinkage
                      && F.getName() != config.entry_point_name && F.getCallingConv() != llvm::CallingConv::PTX_Kernel)
                  {
                    F.setLinkage(llvm::GlobalValue::InternalLinkage);
                    // Remove attributes that conflict with inlining
                    F.removeFnAttr(llvm::Attribute::NoInline);
                    F.removeFnAttr(llvm::Attribute::OptimizeNone);
                    F.addFnAttr(llvm::Attribute::AlwaysInline);
                  }
                }

                llvm::OptimizationLevel opt_level;
                switch (config.optimization_level)
                {
                  case 0:
                    opt_level = llvm::OptimizationLevel::O0;
                    break;
                  case 1:
                    opt_level = llvm::OptimizationLevel::O1;
                    break;
                  case 3:
                    opt_level = llvm::OptimizationLevel::O3;
                    break;
                  default:
                    opt_level = llvm::OptimizationLevel::O2;
                    break;
                }

                // Raise LLVM's loop-unroll thresholds (once) so the user op's
                // small, constant-trip-count loops -- e.g. Numba `local.array`
                // loops -- get FULLY unrolled. Without full unroll the backing
                // alloca keeps a dynamic index, SROA can't promote it, and it
                // lands in local memory (a per-thread stack frame + LDL/STL
                // traffic). ptxas does this promotion on the v1/LTO path; the
                // LLVM-NVPTX path needs full-unroll-then-SROA at the IR level.
                static const bool unroll_tuned = [] {
                  auto& opts   = llvm::cl::getRegisteredOptions();
                  auto set_opt = [&](llvm::StringRef name, llvm::StringRef value) {
                    auto it = opts.find(name);
                    if (it != opts.end())
                    {
                      it->second->addOccurrence(0, name, value);
                    }
                  };
                  set_opt("unroll-threshold", "4000");
                  set_opt("unroll-full-max-count", "1024");
                  set_opt("unroll-max-upperbound", "1024");
                  return true;
                }();
                (void) unroll_tuned;

                llvm::LoopAnalysisManager LAM;
                llvm::FunctionAnalysisManager FAM;
                llvm::CGSCCAnalysisManager CGAM;
                llvm::ModuleAnalysisManager MAM;

                llvm::PassBuilder PB(tm);
                PB.registerModuleAnalyses(MAM);
                PB.registerCGSCCAnalyses(CGAM);
                PB.registerFunctionAnalyses(FAM);
                PB.registerLoopAnalyses(LAM);
                PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

                auto MPM = PB.buildPerModuleDefaultPipeline(opt_level);
                MPM.run(*mod, MAM);

                // Second optimization round with fresh analyses: now that the
                // op's loops are fully unrolled (constant indices), the early
                // SROA in the pipeline promotes the local arrays to registers.
                llvm::LoopAnalysisManager LAM2;
                llvm::FunctionAnalysisManager FAM2;
                llvm::CGSCCAnalysisManager CGAM2;
                llvm::ModuleAnalysisManager MAM2;

                llvm::PassBuilder PB2(tm);
                PB2.registerModuleAnalyses(MAM2);
                PB2.registerCGSCCAnalyses(CGAM2);
                PB2.registerFunctionAnalyses(FAM2);
                PB2.registerLoopAnalyses(LAM2);
                PB2.crossRegisterProxies(LAM2, FAM2, CGAM2, MAM2);

                auto MPM2 = PB2.buildPerModuleDefaultPipeline(opt_level);
                MPM2.run(*mod, MAM2);
              }

              std::error_code EC;
              llvm::raw_fd_ostream dest(output_ptx, EC);
              if (!EC)
              {
                llvm::legacy::PassManager pass;
                tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile);
                pass.run(*mod);
                dest.flush();

                // Debug: when LIBNVCC_DUMP_DIR is set, dump the optimized IR
                // and the PTX fed to ptxas, keyed by entry point name. Lets us
                // inspect codegen (register pressure, launch bounds) post-inline.
                if (const char* dump_dir = std::getenv("LIBNVCC_DUMP_DIR"))
                {
                  std::error_code dec;
                  std::filesystem::create_directories(dump_dir, dec);
                  const std::string base =
                    config.entry_point_name.empty() ? std::string("kernel") : config.entry_point_name;
                  const std::string stem = (std::filesystem::path(dump_dir) / base).string();
                  llvm::raw_fd_ostream ll_os(stem + ".opt.ll", dec);
                  if (!dec)
                  {
                    mod->print(ll_os, nullptr);
                  }
                  std::error_code cec;
                  std::filesystem::copy_file(
                    output_ptx, stem + ".ptx", std::filesystem::copy_options::overwrite_existing, cec);
                  llvm::errs() << "[hostjit] dumped " << stem << ".opt.ll and " << stem << ".ptx\n";
                }
              }
              else
              {
                diagnostics += "Failed to open output file: " + output_ptx + "\n";
                success = false;
              }
            }
            else
            {
              diagnostics += "Failed to create target machine\n";
              success = false;
            }
          }
          else
          {
            diagnostics += "Failed to lookup target: " + err_str + "\n";
            success = false;
          }
        }
      }
    }

    diag_stream.flush();
    diagnostics += diag_output;

    return success;
  }

  BitcodeResult compileToDeviceBitcode(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_bitcode_path,
    const CompilerOptions& config)
  {
    BitcodeResult result;
    result.success = false;

    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_bc_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }

    std::string input_file   = input_name.empty() ? std::string("input.cu") : input_name;
    std::string source_file  = temp_dir + "/" + input_file;
    std::string resource_dir = CLANG_RESOURCE_DIR;

    // PTX version floor is 7.8. Some generated device code uses features
    // added in PTX 7.6 (e.g. `bmsk`), so older versions can fail to assemble
    // even on sm_75/sm_80.
    int ptx_version = 78;
    if (config.sm_version >= 120)
    {
      ptx_version = 87;
    }
    else if (config.sm_version >= 100)
    {
      ptx_version = 85;
    }
    else if (config.sm_version >= 90)
    {
      ptx_version = 80;
    }

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-aux-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-S");
    arg_strings.push_back("-aux-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-is-device");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mlink-builtin-bitcode");
    arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("sm_" + std::to_string(config.sm_version));
    arg_strings.push_back("-target-feature");
    arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("-fcxx-exceptions");
    arg_strings.push_back("-fexceptions");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-Wno-c++11-narrowing");
    arg_strings.push_back("-std=c++17");
    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      result.diagnostics = diag_output + "\nFailed to create compiler invocation";
      removeAll(temp_dir);
      return result;
    }

    if (!config.device_pch_path.empty())
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = config.device_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();

    llvm::LLVMContext llvm_context;
    clang::EmitLLVMOnlyAction emit_llvm_action(&llvm_context);
    bool success = compiler.ExecuteAction(emit_llvm_action);

    if (success)
    {
      std::unique_ptr<llvm::Module> mod = emit_llvm_action.takeModule();
      if (mod)
      {
        std::error_code ec;
        llvm::raw_fd_ostream os(output_bitcode_path, ec, llvm::sys::fs::OF_None);
        if (ec)
        {
          result.diagnostics = "Failed to open bitcode output file: " + output_bitcode_path + "\n";
        }
        else
        {
          llvm::WriteBitcodeToFile(*mod, os);
          os.flush();
          if (os.has_error())
          {
            result.diagnostics = "Failed to write bitcode output file: " + output_bitcode_path + "\n";
          }
          else
          {
            result.success = true;
          }
        }
      }
      else
      {
        result.diagnostics = "Failed to get LLVM module";
      }
    }

    diag_stream.flush();
    result.diagnostics += diag_output;
    if (!config.keep_artifacts)
    {
      removeAll(temp_dir);
    }
    return result;
  }

  bool compileHostCode(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& fatbin_path,
    const std::string& output_obj,
    const CompilerOptions& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_obj).parent_path().string();
    std::string source_file = temp_dir + "/host_" + input_file;

    std::string resource_dir = CLANG_RESOURCE_DIR;

    std::vector<std::string> arg_strings;
    arg_strings.push_back(source_file);
    arg_strings.push_back("-triple");
#ifdef _WIN32
    arg_strings.push_back("x86_64-pc-windows-msvc");
#else
    arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
    arg_strings.push_back("-aux-triple");
    arg_strings.push_back("nvptx64-nvidia-cuda");
    arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
    arg_strings.push_back("-emit-obj");
    arg_strings.push_back("-target-cpu");
    arg_strings.push_back("x86-64");
    arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
    arg_strings.push_back("-fms-compatibility");
    arg_strings.push_back("-fms-compatibility-version=19.40");
    // We do not have access to the windows CRT, but we are only running single threaded anyway
    // Otherwise we have undefined symbols like _tls_index and _Init_thread_epoch
    arg_strings.push_back("-fno-threadsafe-statics");
#else
    arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
    arg_strings.push_back("-mrelocation-model");
    arg_strings.push_back("pic");
    arg_strings.push_back("-pic-level");
    arg_strings.push_back("2");
    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("--offload-new-driver");
    arg_strings.push_back("-fskip-odr-check-in-gmf");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    std::string host_pch_path = config.host_pch_path;

    // Add fatbin embedding (per-build, not part of PCH)
    arg_strings.push_back("-fcuda-include-gpubinary");
    arg_strings.push_back(fatbin_path);

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    if (config.verbose)
    {
      diagnostics += "Host args: ";
      for (const auto& arg : arg_strings)
      {
        diagnostics += arg + " ";
      }
      diagnostics += "\n";
    }

    std::string diag_output;
    llvm::raw_string_ostream diag_stream(diag_output);

    clang::DiagnosticOptions diag_opts;
    diag_opts.ShowColors                       = false;
    clang::TextDiagnosticPrinter* diag_printer = new clang::TextDiagnosticPrinter(diag_stream, diag_opts);
    clang::IntrusiveRefCntPtr<clang::DiagnosticIDs> diag_ids(new clang::DiagnosticIDs());
    clang::DiagnosticsEngine diag_engine(diag_ids, diag_opts, diag_printer);

    clang::CompilerInstance compiler;
    auto& invocation = compiler.getInvocation();

    if (!clang::CompilerInvocation::CreateFromArgs(invocation, args, diag_engine))
    {
      diag_stream.flush();
      diagnostics += diag_output;
      diagnostics += "\nFailed to create host compiler invocation";
      return false;
    }

    // --- PCH: load cached host PCH ---
    if (!host_pch_path.empty() && pathExists(host_pch_path))
    {
      invocation.getPreprocessorOpts().ImplicitPCHInclude = host_pch_path;
    }

    auto vfs = createVFSWithSource(source_code, source_file);
    compiler.createDiagnostics(diag_engine.getClient(), false);
    compiler.setVirtualFileSystem(vfs);
    compiler.createFileManager();
    compiler.getFrontendOpts().OutputFile = output_obj;

    if (config.trace_includes)
    {
      diagnostics += "\n=== Host Header Search Paths ===\n";
      const auto& hso = invocation.getHeaderSearchOpts();
      for (const auto& entry : hso.UserEntries)
      {
        diagnostics += "  " + entry.Path + "\n";
      }
      diagnostics += "=== End Header Search Paths ===\n\n";
    }

    clang::EmitObjAction emit_action;
    bool success = compiler.ExecuteAction(emit_action);

    if (config.trace_includes && compiler.hasSourceManager())
    {
      diagnostics += "\n=== Host Included Files ===\n";
      auto& sm = compiler.getSourceManager();
      for (auto it = sm.fileinfo_begin(); it != sm.fileinfo_end(); ++it)
      {
        diagnostics += "  " + it->first.getName().str() + "\n";
      }
      diagnostics += "=== End Included Files ===\n\n";
    }

    diag_stream.flush();
    diagnostics += diag_output;

    return success;
  }

  CompilationResult compileToObject(
    const std::string& source_code,
    const std::string& input_name,
    const std::string& output_path,
    const std::string& output_cubin_path,
    const CompilerOptions& config)
  {
    CompilationResult result;
    result.success          = false;
    result.object_file_path = output_path;

    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (tempDirectoryPath() / ("hostjit_" + std::to_string(reinterpret_cast<uintptr_t>(this)))).string();
    if (!createDirectories(temp_dir, result.diagnostics))
    {
      return result;
    }

    std::string input_file  = input_name.empty() ? std::string("input.cu") : input_name;
    std::string ptx_file    = temp_dir + "/device.ptx";
    std::string fatbin_file = temp_dir + "/device.fatbin";

    if (config.verbose)
    {
      result.diagnostics += "=== Device compilation ===\n";
    }

    if (!compileDeviceToPTX(source_code, input_file, ptx_file, config, result.diagnostics))
    {
      result.diagnostics += "\nDevice compilation failed";
      removeAll(temp_dir);
      return result;
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== nvJitLink + fatbinary ===\n";
    }

    {
      std::vector<char> ptx_data;
      {
        std::ifstream f(ptx_file, std::ios::binary);
        ptx_data.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
      }
      if (ptx_data.empty())
      {
        result.diagnostics += "\nFailed to read ptx file";
        removeAll(temp_dir);
        return result;
      }
      if (ptx_data.back() != '\0')
      {
        ptx_data.push_back('\0');
      }

      std::string arch_opt  = "-arch=sm_" + std::to_string(config.sm_version);
      std::string opt_level = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
      std::vector<std::string> jitlink_option_strs{arch_opt, opt_level};
      // LTOIR inputs require -lto. When present, both the PTX and the LTOIRs
      // get linked through the LTO codegen path.
      const bool have_ltoir = !config.device_ltoir_files.empty();
      if (have_ltoir)
      {
        jitlink_option_strs.emplace_back("-lto");
      }
      std::vector<const char*> jitlink_options;
      jitlink_options.reserve(jitlink_option_strs.size());
      for (const auto& s : jitlink_option_strs)
      {
        jitlink_options.push_back(s.c_str());
      }

      nvJitLinkHandle jitlink_handle = nullptr;
      nvJitLinkResult jlr =
        nvJitLinkCreate(&jitlink_handle, static_cast<uint32_t>(jitlink_options.size()), jitlink_options.data());
      if (jlr != NVJITLINK_SUCCESS)
      {
        result.diagnostics += "\nnvJitLinkCreate failed (error " + std::to_string(static_cast<int>(jlr)) + ")";
        result.diagnostics += "\nnvJitLink options:";
        for (const auto& option : jitlink_option_strs)
        {
          result.diagnostics += " " + option;
        }
        removeAll(temp_dir);
        return result;
      }

      jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_PTX, ptx_data.data(), ptx_data.size(), "device.ptx");
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += "\nnvJitLinkAddData failed";
        nvJitLinkDestroy(&jitlink_handle);
        removeAll(temp_dir);
        return result;
      }

      // Feed LTO-IR inputs to nvJitLink alongside the device PTX. This is the
      // escape-hatch path for callers with pre-built nvcc -dlto artifacts;
      // Python-emitted user ops travel as LLVM bitcode through the path above
      // and are already inlined into the PTX by the time we get here.
      // nvJitLink resolves any remaining extern symbol(s) from these modules.
      for (const auto& ltoir_path : config.device_ltoir_files)
      {
        std::ifstream f(ltoir_path, std::ios::binary);
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.empty())
        {
          continue;
        }
        jlr = nvJitLinkAddData(jitlink_handle, NVJITLINK_INPUT_LTOIR, buf.data(), buf.size(), ltoir_path.c_str());
        if (jlr != NVJITLINK_SUCCESS)
        {
          size_t log_size = 0;
          nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
          if (log_size > 1)
          {
            std::string log(log_size, '\0');
            nvJitLinkGetErrorLog(jitlink_handle, log.data());
            result.diagnostics += "\n" + log;
          }
          result.diagnostics += "\nnvJitLinkAddData(LTOIR) failed for " + ltoir_path;
          nvJitLinkDestroy(&jitlink_handle);
          removeAll(temp_dir);
          return result;
        }
      }

      jlr = nvJitLinkComplete(jitlink_handle);
      if (jlr != NVJITLINK_SUCCESS)
      {
        size_t log_size = 0;
        nvJitLinkGetErrorLogSize(jitlink_handle, &log_size);
        if (log_size > 1)
        {
          std::string log(log_size, '\0');
          nvJitLinkGetErrorLog(jitlink_handle, log.data());
          result.diagnostics += "\n" + log;
        }
        result.diagnostics += "\nnvJitLinkComplete failed";
        nvJitLinkDestroy(&jitlink_handle);
        removeAll(temp_dir);
        return result;
      }

      size_t cubin_size = 0;
      nvJitLinkGetLinkedCubinSize(jitlink_handle, &cubin_size);
      std::vector<char> cubin_data(cubin_size);
      nvJitLinkGetLinkedCubin(jitlink_handle, cubin_data.data());
      nvJitLinkDestroy(&jitlink_handle);

      if (!output_cubin_path.empty())
      {
        std::ofstream cubin_out(output_cubin_path, std::ios::binary);
        cubin_out.write(cubin_data.data(), static_cast<std::streamsize>(cubin_data.size()));
        if (!cubin_out)
        {
          result.diagnostics += "\nFailed to write cubin file";
          removeAll(temp_dir);
          return result;
        }
      }

      std::string arch             = std::to_string(config.sm_version);
      const char* fatbin_options[] = {"-64", "-cuda"};
      nvFatbinHandle fatbin_handle = nullptr;
      nvFatbinResult fbr           = nvFatbinCreate(&fatbin_handle, fatbin_options, 2);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinCreate failed: ") + nvFatbinGetErrorString(fbr);
        removeAll(temp_dir);
        return result;
      }

      fbr = nvFatbinAddCubin(fatbin_handle, cubin_data.data(), cubin_data.size(), arch.c_str(), "device.cubin");
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        removeAll(temp_dir);
        return result;
      }

      fbr = nvFatbinAddPTX(fatbin_handle, ptx_data.data(), ptx_data.size(), arch.c_str(), "device.ptx", nullptr);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddPTX failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        removeAll(temp_dir);
        return result;
      }

      size_t fatbin_size = 0;
      fbr                = nvFatbinSize(fatbin_handle, &fatbin_size);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinSize failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        removeAll(temp_dir);
        return result;
      }

      std::vector<char> fatbin_data(fatbin_size);
      fbr = nvFatbinGet(fatbin_handle, fatbin_data.data());
      nvFatbinDestroy(&fatbin_handle);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinGet failed: ") + nvFatbinGetErrorString(fbr);
        removeAll(temp_dir);
        return result;
      }

      std::ofstream out(fatbin_file, std::ios::binary);
      out.write(fatbin_data.data(), static_cast<std::streamsize>(fatbin_data.size()));
      if (!out)
      {
        result.diagnostics += "\nFailed to write fatbin file";
        removeAll(temp_dir);
        return result;
      }
    }

    if (config.verbose)
    {
      result.diagnostics += "\n=== Host compilation ===\n";
    }

    if (!compileHostCode(source_code, input_file, fatbin_file, output_path, config, result.diagnostics))
    {
      result.diagnostics += "\nHost compilation failed";
      removeAll(temp_dir);
      return result;
    }

    if (!config.keep_artifacts)
    {
      removeAll(temp_dir);
    }
    result.success = true;
    return result;
  }

  bool createPCH(const std::string& source_code,
                 libnvccPCHKind kind,
                 const std::string& pch_source_path,
                 const std::string& pch_output_path,
                 const CompilerOptions& config,
                 std::string& diagnostics)
  {
    std::string error_msg;
    if (!validateOptions(config, &error_msg))
    {
      diagnostics = "Configuration error: " + error_msg;
      return false;
    }

    initialize_llvm();

    std::string resource_dir = CLANG_RESOURCE_DIR;
    std::vector<std::string> arg_strings;
    arg_strings.push_back(pch_source_path);

    if (kind == LIBNVCC_PCH_DEVICE)
    {
      int ptx_version = 78;
      if (config.sm_version >= 120)
      {
        ptx_version = 87;
      }
      else if (config.sm_version >= 100)
      {
        ptx_version = 85;
      }
      else if (config.sm_version >= 90)
      {
        ptx_version = 80;
      }

      arg_strings.push_back("-triple");
      arg_strings.push_back("nvptx64-nvidia-cuda");
      arg_strings.push_back("-aux-triple");
#ifdef _WIN32
      arg_strings.push_back("x86_64-pc-windows-msvc");
#else
      arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
      arg_strings.push_back("-S");
      arg_strings.push_back("-aux-target-cpu");
      arg_strings.push_back("x86-64");
      arg_strings.push_back("-fcuda-is-device");
      arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
      arg_strings.push_back("-fms-compatibility");
      arg_strings.push_back("-fms-compatibility-version=19.40");
#else
      arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
      arg_strings.push_back("-mlink-builtin-bitcode");
      arg_strings.push_back(config.cuda_toolkit_path + "/nvvm/libdevice/libdevice.10.bc");
      arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
      arg_strings.push_back("-target-cpu");
      arg_strings.push_back("sm_" + std::to_string(config.sm_version));
      arg_strings.push_back("-target-feature");
      arg_strings.push_back("+ptx" + std::to_string(ptx_version));
    }
    else if (kind == LIBNVCC_PCH_HOST)
    {
      arg_strings.push_back("-triple");
#ifdef _WIN32
      arg_strings.push_back("x86_64-pc-windows-msvc");
#else
      arg_strings.push_back("x86_64-pc-linux-gnu");
#endif
      arg_strings.push_back("-aux-triple");
      arg_strings.push_back("nvptx64-nvidia-cuda");
      arg_strings.push_back("-target-sdk-version=" CUDA_SDK_VERSION);
      arg_strings.push_back("-emit-obj");
      arg_strings.push_back("-target-cpu");
      arg_strings.push_back("x86-64");
      arg_strings.push_back("-fcuda-allow-variadic-functions");
#ifdef _WIN32
      arg_strings.push_back("-fms-compatibility");
      arg_strings.push_back("-fms-compatibility-version=19.40");
#else
      arg_strings.push_back("-fgnuc-version=4.2.1");
#endif
      arg_strings.push_back("-mrelocation-model");
      arg_strings.push_back("pic");
      arg_strings.push_back("-pic-level");
      arg_strings.push_back("2");
    }
    else
    {
      diagnostics = "Invalid PCH kind";
      return false;
    }

    arg_strings.push_back("-resource-dir");
    arg_strings.push_back(resource_dir);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/stubs");
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(
      config.clang_headers_path.empty() ? std::string(CLANG_HEADERS_DIR) : config.clang_headers_path);
    appendSystemIncludePaths(arg_strings, config);
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    appendIncludePaths(arg_strings, config);

    if (kind == LIBNVCC_PCH_DEVICE)
    {
      arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    }
    arg_strings.push_back("-DNDEBUG");

    appendMacroDefinitions(arg_strings, config);

    arg_strings.push_back("-fdeprecated-macro");
    if (kind == LIBNVCC_PCH_DEVICE)
    {
      arg_strings.push_back("--offload-new-driver");
      arg_strings.push_back("-fskip-odr-check-in-gmf");
      arg_strings.push_back("-fcxx-exceptions");
      arg_strings.push_back("-fexceptions");
    }
    else
    {
      arg_strings.push_back("--offload-new-driver");
      arg_strings.push_back("-fskip-odr-check-in-gmf");
    }
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    appendExtraClangArgs(arg_strings, config);
    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    return generatePCH(source_code, pch_source_path, pch_output_path, arg_strings, diagnostics);
  }

  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& object_files, const std::string& output_path, const CompilerOptions& config)
  {
    LinkResult result;
    result.success      = false;
    result.library_path = output_path;

    if (object_files.empty())
    {
      result.diagnostics = "No object files provided";
      return result;
    }

    std::vector<std::string> arg_strings;

#ifdef _WIN32
    arg_strings.push_back("lld-link");
    arg_strings.push_back("/DLL");
    arg_strings.push_back("/NOENTRY");
    arg_strings.push_back("/NODEFAULTLIB");
    arg_strings.push_back("/OUT:" + output_path);

    // Generate import libraries from DLLs present on the system,
    // so we don't require the Windows SDK or MSVC .lib files.
    std::string implib_dir = std::filesystem::path(output_path).parent_path().string();

    std::string cudart_dll = findCudartDllName(config.cuda_toolkit_path);
    generateImportLib(
      cudart_dll,
      {"cudaMalloc",
       "cudaFree",
       "cudaMemcpy",
       "cudaMemcpyAsync",
       "cudaMemset",
       "cudaMemsetAsync",
       "cudaDeviceSynchronize",
       "cudaFuncSetAttribute",
       "cudaGetDevice",
       "cudaGetDeviceProperties",
       "cudaGetLastError",
       "cudaPeekAtLastError",
       "cudaGetErrorString",
       "cudaStreamCreate",
       "cudaStreamDestroy",
       "cudaStreamSynchronize",
       "cudaEventCreate",
       "cudaEventDestroy",
       "cudaEventRecord",
       "cudaEventSynchronize",
       "cudaEventElapsedTime",
       "cudaMallocAsync",
       "cudaFreeAsync",
       "cudaDeviceGetAttribute",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
       "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
       "cudaFuncGetAttributes",
       "cudaLaunchKernel",
       "cudaLaunchKernelExC",
       "__cudaRegisterFatBinary",
       "__cudaRegisterFatBinaryEnd",
       "__cudaUnregisterFatBinary",
       "__cudaRegisterFunction",
       "__cudaRegisterVar",
       "__cudaPushCallConfiguration",
       "__cudaPopCallConfiguration"},
      implib_dir + "/cudart.lib");

    generateImportLib(
      "ucrtbase.dll",
      {"malloc",
       "free",
       "calloc",
       "realloc",
       "_callnewh",
       "_errno",
       "abort",
       "exit",
       "_exit",
       "_register_onexit_function",
       "_crt_atexit",
       "_initterm",
       "_initterm_e",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "strlen",
       "strcmp",
       "strncmp",
       "_initialize_onexit_table",
       "_execute_onexit_table",
       "_register_thread_local_exe_atexit_callback"},
      implib_dir + "/ucrt.lib");

    generateImportLib(
      "vcruntime140.dll",
      {"__std_exception_copy",
       "__std_exception_destroy",
       "__CxxFrameHandler3",
       "_CxxThrowException",
       "memcpy",
       "memset",
       "memmove",
       "memcmp",
       "__std_type_info_destroy_list",
       "_purecall"},
      implib_dir + "/vcruntime.lib");

    generateImportLib(
      "kernel32.dll",
      {"InitializeCriticalSection",
       "EnterCriticalSection",
       "LeaveCriticalSection",
       "DeleteCriticalSection",
       "InitOnceExecuteOnce",
       "LoadLibraryExA",
       "LoadLibraryExW",
       "GetProcAddress",
       "FreeLibrary",
       "GetModuleHandleA",
       "GetLastError",
       "SetLastError",
       "GetCurrentProcess",
       "GetCurrentThread",
       "GetCurrentThreadId",
       "VirtualProtect",
       "FlushInstructionCache",
       "QueryPerformanceCounter",
       "QueryPerformanceFrequency"},
      implib_dir + "/kernel32.lib");

    arg_strings.push_back("/LIBPATH:" + implib_dir);

    for (const auto& obj_file : object_files)
    {
      arg_strings.push_back(obj_file);
    }

    arg_strings.push_back("cudart.lib");
    arg_strings.push_back("ucrt.lib");
    arg_strings.push_back("vcruntime.lib");
    arg_strings.push_back("kernel32.lib");
#else
    arg_strings.push_back("ld.lld");
    arg_strings.push_back("-shared");
    arg_strings.push_back("--build-id");
    arg_strings.push_back("--eh-frame-hdr");
    arg_strings.push_back("-m");
    arg_strings.push_back("elf_x86_64");
    // Allow unresolved symbols — they will be satisfied at dlopen() time
    // by libraries already loaded in the host process (libc, libstdc++,
    // cudart, etc.).  This removes the need for system CRT objects and
    // dev packages on the target machine.
    arg_strings.push_back("--allow-shlib-undefined");
    arg_strings.push_back("-o");
    arg_strings.push_back(output_path);

    for (const auto& lib_path : config.library_paths)
    {
      arg_strings.push_back("-L" + lib_path);
      // Embed the library path as RPATH so the dynamic linker can find
      // libcudart.so.XX at dlopen time without LD_LIBRARY_PATH.
      arg_strings.push_back("-rpath");
      arg_strings.push_back(lib_path);
    }

    for (const auto& obj_file : object_files)
    {
      arg_strings.push_back(obj_file);
    }

    // pip packages ship libcudart.so.XX without an unversioned symlink,
    // so -lcudart won't work.  Find the actual .so by scanning library_paths.
    {
      bool found_cudart = false;
      for (const auto& lib_path : config.library_paths)
      {
        if (!pathExists(lib_path))
        {
          continue;
        }
        forEachDirectoryEntry(lib_path, [&](const std::filesystem::directory_entry& entry) {
          auto fname = entry.path().filename().string();
          if (!found_cudart && fname.starts_with("libcudart.so"))
          {
            arg_strings.push_back(entry.path().string());
            found_cudart = true;
          }
        });
        if (found_cudart)
        {
          break;
        }
      }
      if (!found_cudart)
      {
        arg_strings.push_back("-lcudart");
      }
    }
#endif

    std::vector<const char*> args;
    for (const auto& arg : arg_strings)
    {
      args.push_back(arg.c_str());
    }

    std::string stdout_str, stderr_str;
    llvm::raw_string_ostream stdout_os(stdout_str);
    llvm::raw_string_ostream stderr_os(stderr_str);

#ifdef _WIN32
    bool link_success = lld::coff::link(args, stdout_os, stderr_os, false, false);
#else
    bool link_success = lld::elf::link(args, stdout_os, stderr_os, false, false);
#endif

    stdout_os.flush();
    stderr_os.flush();

    if (!stdout_str.empty())
    {
      result.diagnostics += stdout_str;
    }
    if (!stderr_str.empty())
    {
      result.diagnostics += stderr_str;
    }

    if (!link_success)
    {
      result.diagnostics += "\nLinking failed";
      return result;
    }

    result.success = true;
    return result;
  }
};
} // namespace libnvcc

struct libnvccProgram_st
{
  std::string source;
  std::string name;
  std::string log;
  libnvcc::CompilerImpl compiler;
};

namespace
{
void setProgramLog(libnvccProgram prog, std::string log)
{
  if (prog)
  {
    prog->log = std::move(log);
  }
}

bool parseProgramOptions(
  libnvccProgram prog, int num_options, const char* const* raw_options, libnvcc::CompilerOptions& options)
{
  std::string error;
  if (!libnvcc::parseOptions(num_options, raw_options, options, error))
  {
    setProgramLog(prog, "Option error: " + error);
    return false;
  }
  return true;
}
} // anonymous namespace

extern "C" const char* libnvccGetErrorString(libnvccResult result)
{
  switch (result)
  {
    case LIBNVCC_SUCCESS:
      return "LIBNVCC_SUCCESS";
    case LIBNVCC_ERROR_OUT_OF_MEMORY:
      return "LIBNVCC_ERROR_OUT_OF_MEMORY";
    case LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE:
      return "LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE";
    case LIBNVCC_ERROR_INVALID_INPUT:
      return "LIBNVCC_ERROR_INVALID_INPUT";
    case LIBNVCC_ERROR_INVALID_PROGRAM:
      return "LIBNVCC_ERROR_INVALID_PROGRAM";
    case LIBNVCC_ERROR_INVALID_OPTION:
      return "LIBNVCC_ERROR_INVALID_OPTION";
    case LIBNVCC_ERROR_COMPILATION:
      return "LIBNVCC_ERROR_COMPILATION";
    case LIBNVCC_ERROR_LINKING:
      return "LIBNVCC_ERROR_LINKING";
    case LIBNVCC_ERROR_PCH_CREATE:
      return "LIBNVCC_ERROR_PCH_CREATE";
    case LIBNVCC_ERROR_INTERNAL_ERROR:
      return "LIBNVCC_ERROR_INTERNAL_ERROR";
  }
  return "LIBNVCC_ERROR_UNKNOWN";
}

extern "C" libnvccResult libnvccCreateProgram(libnvccProgram* prog, const char* src, const char* name)
{
  if (!prog || !src)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *prog = nullptr;

  auto* program   = new libnvccProgram_st;
  program->source = src;
  program->name   = (name && name[0]) ? name : "input.cu";
  *prog           = program;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccDestroyProgram(libnvccProgram* prog)
{
  if (!prog || !*prog)
  {
    return LIBNVCC_SUCCESS;
  }
  delete *prog;
  *prog = nullptr;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccCompileProgramToDeviceBitcode(
  libnvccProgram prog, const char* outputBitcodePath, int numOptions, const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputBitcodePath || outputBitcodePath[0] == '\0')
  {
    setProgramLog(prog, "outputBitcodePath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  auto result = prog->compiler.compileToDeviceBitcode(prog->source, prog->name, outputBitcodePath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccCompileProgramToObject(
  libnvccProgram prog,
  const char* outputObjectPath,
  const char* outputCubinPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!outputObjectPath || outputObjectPath[0] == '\0')
  {
    setProgramLog(prog, "outputObjectPath must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  const std::string cubin_path = outputCubinPath ? outputCubinPath : "";
  auto result = prog->compiler.compileToObject(prog->source, prog->name, outputObjectPath, cubin_path, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_COMPILATION;
}

extern "C" libnvccResult libnvccLinkToSharedLibrary(
  libnvccProgram prog,
  int numObjectFiles,
  const char* const* objectFiles,
  const char* outputLibraryPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (numObjectFiles < 0 || (numObjectFiles > 0 && !objectFiles) || !outputLibraryPath || outputLibraryPath[0] == '\0')
  {
    setProgramLog(prog, "Invalid link input");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::vector<std::string> object_files;
  object_files.reserve(static_cast<size_t>(numObjectFiles));
  for (int i = 0; i < numObjectFiles; ++i)
  {
    if (!objectFiles[i] || objectFiles[i][0] == '\0')
    {
      setProgramLog(prog, "Object file path must be non-empty");
      return LIBNVCC_ERROR_INVALID_INPUT;
    }
    object_files.emplace_back(objectFiles[i]);
  }

  auto result = prog->compiler.linkToSharedLibrary(object_files, outputLibraryPath, parsed_options);
  setProgramLog(prog, result.diagnostics);
  return result.success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_LINKING;
}

extern "C" libnvccResult libnvccCreatePCH(
  libnvccProgram prog,
  libnvccPCHKind kind,
  const char* pchSourcePath,
  const char* pchOutputPath,
  int numOptions,
  const char* const* options)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!pchSourcePath || pchSourcePath[0] == '\0' || !pchOutputPath || pchOutputPath[0] == '\0')
  {
    setProgramLog(prog, "PCH source and output paths must be non-empty");
    return LIBNVCC_ERROR_INVALID_INPUT;
  }

  libnvcc::CompilerOptions parsed_options;
  if (!parseProgramOptions(prog, numOptions, options, parsed_options))
  {
    return LIBNVCC_ERROR_INVALID_OPTION;
  }

  std::string diagnostics;
  bool success =
    prog->compiler.createPCH(prog->source, kind, pchSourcePath, pchOutputPath, parsed_options, diagnostics);
  setProgramLog(prog, diagnostics);
  return success ? LIBNVCC_SUCCESS : LIBNVCC_ERROR_PCH_CREATE;
}

extern "C" libnvccResult libnvccGetProgramLogSize(libnvccProgram prog, size_t* logSizeRet)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!logSizeRet)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  *logSizeRet = prog->log.size() + 1;
  return LIBNVCC_SUCCESS;
}

extern "C" libnvccResult libnvccGetProgramLog(libnvccProgram prog, char* log)
{
  if (!prog)
  {
    return LIBNVCC_ERROR_INVALID_PROGRAM;
  }
  if (!log)
  {
    return LIBNVCC_ERROR_INVALID_INPUT;
  }
  std::memcpy(log, prog->log.c_str(), prog->log.size() + 1);
  return LIBNVCC_SUCCESS;
}
