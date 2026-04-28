#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendOptions.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <hostjit/compiler.hpp>
#include <hostjit/config.hpp>
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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <vector>

#include <nvFatbin.h>
#include <nvJitLink.h>

namespace hostjit
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
    if (!fs::exists(dir))
    {
      continue;
    }
    for (const auto& entry : fs::directory_iterator(dir))
    {
      auto name = entry.path().filename().string();
      if (name.starts_with("cudart64_") && name.ends_with(".dll"))
      {
        return name;
      }
    }
  }
  return "cudart64_12.dll"; // fallback
}
#endif

// Headers precompiled into the PCH cache.  Covers the algorithms exposed
// by the C parallel library so that a single pair of PCH files (device +
// host) is reused across reduce, adjacent-difference, etc.
static constexpr const char* pch_preamble_source =
  "#include <cuda_runtime.h>\n"
  "#include <cuda/std/iterator>\n"
  "#include <cuda/std/functional>\n"
  "#include <cuda/functional>\n"
  "#include <cub/device/device_reduce.cuh>\n"
  "#include <cub/device/device_adjacent_difference.cuh>\n";

class CUDACompiler::Impl
{
public:
  Impl() {}

  // Get the persistent PCH cache directory.
  static std::filesystem::path getPCHCacheDir()
  {
    auto dir = std::filesystem::temp_directory_path() / "hostjit_pch";
    std::filesystem::create_directories(dir);
    return dir;
  }

  // Get a persistent cache path for a PCH file.
  static std::string getPCHPath(const std::string& kind, int sm_version)
  {
    return (getPCHCacheDir() / (kind + "_sm" + std::to_string(sm_version) + ".pch")).string();
  }

  // Get the persistent path for the PCH preamble source file.
  // The PCH stores a reference to this path, so it must be stable across runs.
  static std::string getPCHSourcePath(const std::string& kind, int sm_version)
  {
    return (getPCHCacheDir() / (kind + "_sm" + std::to_string(sm_version) + "_preamble.cu")).string();
  }

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
    const CompilerConfig& config,
    std::string& diagnostics)
  {
    std::string temp_dir    = std::filesystem::path(output_ptx).parent_path().string();
    std::string source_file = temp_dir + "/" + input_file;

    std::string resource_dir = CLANG_RESOURCE_DIR;

    int ptx_version = 70;
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
    else if (config.sm_version >= 89)
    {
      ptx_version = 78;
    }
    else if (config.sm_version >= 80)
    {
      ptx_version = 75;
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
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include/cuda/std");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path + "/cuda/std");
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/cub");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/thrust");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    for (const auto& include_path : config.include_paths)
    {
      arg_strings.push_back("-I" + include_path);
    }

    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");
    arg_strings.push_back("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK");
    arg_strings.push_back("-DCCCL_DISABLE_FP16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_BF16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_NVTX=1");
    arg_strings.push_back("-DCCCL_DISABLE_EXCEPTIONS=1");

    std::vector<std::string> bitcode_files_to_link = config.device_bitcode_files;

    for (const auto& [macro_name, macro_value] : config.macro_definitions)
    {
      if (macro_value.empty())
      {
        arg_strings.push_back("-D" + macro_name);
      }
      else
      {
        arg_strings.push_back("-D" + macro_name + "=" + macro_value);
      }
    }

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

    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    // --- PCH: ensure device PCH exists ---
    std::string device_pch_path;
    if (config.enable_pch)
    {
      device_pch_path = getPCHPath("device", config.sm_version);
      if (!std::filesystem::exists(device_pch_path))
      {
        auto pch_src_path = getPCHSourcePath("device", config.sm_version);
        std::string pch_diag;
        if (!generatePCH(pch_preamble_source, pch_src_path, device_pch_path, arg_strings, pch_diag))
        {
          diagnostics += "Device PCH generation failed: " + pch_diag + "\n";
          device_pch_path.clear();
        }
        else if (config.verbose)
        {
          diagnostics += "Generated device PCH: " + device_pch_path + "\n";
        }
      }
    }

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
    if (!device_pch_path.empty() && std::filesystem::exists(device_pch_path))
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
            diagnostics += "Failed to parse bitcode: " + bc_file + "\n";
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
              }

              std::error_code EC;
              llvm::raw_fd_ostream dest(output_ptx, EC);
              if (!EC)
              {
                llvm::legacy::PassManager pass;
                tm->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::AssemblyFile);
                pass.run(*mod);
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

  BitcodeResult compileToDeviceBitcode(const std::string& source_code, const CompilerConfig& config)
  {
    BitcodeResult result;
    result.success = false;

    std::string error_msg;
    if (!validateConfig(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (std::filesystem::temp_directory_path() / ("hostjit_bc_" + std::to_string(reinterpret_cast<uintptr_t>(this))))
        .string();
    std::filesystem::create_directories(temp_dir);

    std::string input_file   = "input.cu";
    std::string source_file  = temp_dir + "/" + input_file;
    std::string resource_dir = CLANG_RESOURCE_DIR;

    int ptx_version = 70;
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
    else if (config.sm_version >= 89)
    {
      ptx_version = 78;
    }
    else if (config.sm_version >= 80)
    {
      ptx_version = 75;
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
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include/cuda/std");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path + "/cuda/std");
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/cub");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/thrust");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");
    arg_strings.push_back("-D__HOSTJIT_DEVICE_COMPILATION__=1");
    arg_strings.push_back("-DNDEBUG");
    arg_strings.push_back("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK");
    arg_strings.push_back("-DCCCL_DISABLE_FP16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_BF16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_NVTX=1");
    arg_strings.push_back("-DCCCL_DISABLE_EXCEPTIONS=1");
    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("-fcxx-exceptions");
    arg_strings.push_back("-fexceptions");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-Wno-c++11-narrowing");
    arg_strings.push_back("-std=c++17");
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
      std::filesystem::remove_all(temp_dir);
      return result;
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
        llvm::SmallVector<char, 0> buffer;
        llvm::raw_svector_ostream os(buffer);
        llvm::WriteBitcodeToFile(*mod, os);
        result.bitcode = std::string(buffer.begin(), buffer.end());
        result.success = true;
      }
      else
      {
        result.diagnostics = "Failed to get LLVM module";
      }
    }

    diag_stream.flush();
    result.diagnostics += diag_output;
    std::filesystem::remove_all(temp_dir);
    return result;
  }

  bool compileHostCode(
    const std::string& source_code,
    const std::string& input_file,
    const std::string& fatbin_path,
    const std::string& output_obj,
    const CompilerConfig& config,
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
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include/cuda/std");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path + "/cuda/std");
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/libcudacxx/include");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/cub");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    if (config.cccl_include_path.empty())
    {
      arg_strings.push_back(std::string(CCCL_SOURCE_DIR) + "/thrust");
    }
    else
    {
      arg_strings.push_back(config.cccl_include_path);
    }
    arg_strings.push_back("-internal-isystem");
    arg_strings.push_back(config.cuda_toolkit_path + "/include");
    arg_strings.push_back("-include");
    arg_strings.push_back(config.hostjit_include_path + "/hostjit/cuda_minimal/__clang_cuda_runtime_wrapper.h");

    for (const auto& include_path : config.include_paths)
    {
      arg_strings.push_back("-I" + include_path);
    }

    arg_strings.push_back("-DNDEBUG");
    arg_strings.push_back("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK");
    arg_strings.push_back("-D_CCCL_ENABLE_FREESTANDING=1");
    arg_strings.push_back("-DCCCL_DISABLE_FP16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_BF16_SUPPORT=1");
    arg_strings.push_back("-DCCCL_DISABLE_NVTX=1");
    arg_strings.push_back("-DCCCL_DISABLE_EXCEPTIONS=1");

    for (const auto& [macro_name, macro_value] : config.macro_definitions)
    {
      if (macro_value.empty())
      {
        arg_strings.push_back("-D" + macro_name);
      }
      else
      {
        arg_strings.push_back("-D" + macro_name + "=" + macro_value);
      }
    }

    arg_strings.push_back("-fdeprecated-macro");
    arg_strings.push_back("--offload-new-driver");
    arg_strings.push_back("-fskip-odr-check-in-gmf");
    arg_strings.push_back("-O" + std::to_string(config.optimization_level));
    arg_strings.push_back("-std=c++17");

    if (config.trace_includes)
    {
      arg_strings.push_back("-H");
    }

    arg_strings.push_back("-x");
    arg_strings.push_back("cuda");

    // --- PCH: ensure host PCH exists (before adding fatbin-specific args) ---
    std::string host_pch_path;
    if (config.enable_pch)
    {
      host_pch_path = getPCHPath("host", config.sm_version);
      if (!std::filesystem::exists(host_pch_path))
      {
        auto pch_src_path = getPCHSourcePath("host", config.sm_version);
        std::string pch_diag;
        if (!generatePCH(pch_preamble_source, pch_src_path, host_pch_path, arg_strings, pch_diag))
        {
          diagnostics += "Host PCH generation failed: " + pch_diag + "\n";
          host_pch_path.clear();
        }
        else if (config.verbose)
        {
          diagnostics += "Generated host PCH: " + host_pch_path + "\n";
        }
      }
    }

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
    if (!host_pch_path.empty() && std::filesystem::exists(host_pch_path))
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

  CompilationResult
  compileToObject(const std::string& source_code, const std::string& output_path, const CompilerConfig& config)
  {
    CompilationResult result;
    result.success          = false;
    result.object_file_path = output_path;

    std::string error_msg;
    if (!validateConfig(config, &error_msg))
    {
      result.diagnostics = "Configuration error: " + error_msg;
      return result;
    }

    initialize_llvm();

    std::string temp_dir =
      (std::filesystem::temp_directory_path() / ("hostjit_" + std::to_string(reinterpret_cast<uintptr_t>(this))))
        .string();
    std::filesystem::create_directories(temp_dir);

    std::string input_file  = "input.cu";
    std::string ptx_file    = temp_dir + "/device.ptx";
    std::string fatbin_file = temp_dir + "/device.fatbin";

    if (config.verbose)
    {
      result.diagnostics += "=== Device compilation ===\n";
    }

    if (!compileDeviceToPTX(source_code, input_file, ptx_file, config, result.diagnostics))
    {
      result.diagnostics += "\nDevice compilation failed";
      std::filesystem::remove_all(temp_dir);
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
        std::filesystem::remove_all(temp_dir);
        return result;
      }
      if (ptx_data.back() != '\0')
      {
        ptx_data.push_back('\0');
      }

      std::string arch_opt           = "-arch=sm_" + std::to_string(config.sm_version);
      std::string opt_level          = "-O" + std::to_string(config.optimization_level >= 1 ? 3 : 0);
      const char* jitlink_options[]  = {arch_opt.c_str(), opt_level.c_str()};
      nvJitLinkHandle jitlink_handle = nullptr;
      nvJitLinkResult jlr            = nvJitLinkCreate(&jitlink_handle, 2, jitlink_options);
      if (jlr != NVJITLINK_SUCCESS)
      {
        result.diagnostics += "\nnvJitLinkCreate failed (error " + std::to_string(static_cast<int>(jlr)) + ")";
        std::filesystem::remove_all(temp_dir);
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
        std::filesystem::remove_all(temp_dir);
        return result;
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
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      size_t cubin_size = 0;
      nvJitLinkGetLinkedCubinSize(jitlink_handle, &cubin_size);
      std::vector<char> cubin_data(cubin_size);
      nvJitLinkGetLinkedCubin(jitlink_handle, cubin_data.data());
      nvJitLinkDestroy(&jitlink_handle);

      // Store cubin in the result for inspection
      result.cubin = cubin_data;

      std::string arch             = std::to_string(config.sm_version);
      const char* fatbin_options[] = {"-64", "-cuda"};
      nvFatbinHandle fatbin_handle = nullptr;
      nvFatbinResult fbr           = nvFatbinCreate(&fatbin_handle, fatbin_options, 2);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinCreate failed: ") + nvFatbinGetErrorString(fbr);
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      fbr = nvFatbinAddCubin(fatbin_handle, cubin_data.data(), cubin_data.size(), arch.c_str(), "device.cubin");
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddCubin failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      fbr = nvFatbinAddPTX(fatbin_handle, ptx_data.data(), ptx_data.size(), arch.c_str(), "device.ptx", nullptr);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinAddPTX failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      size_t fatbin_size = 0;
      fbr                = nvFatbinSize(fatbin_handle, &fatbin_size);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinSize failed: ") + nvFatbinGetErrorString(fbr);
        nvFatbinDestroy(&fatbin_handle);
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      std::vector<char> fatbin_data(fatbin_size);
      fbr = nvFatbinGet(fatbin_handle, fatbin_data.data());
      nvFatbinDestroy(&fatbin_handle);
      if (fbr != NVFATBIN_SUCCESS)
      {
        result.diagnostics += std::string("\nnvFatbinGet failed: ") + nvFatbinGetErrorString(fbr);
        std::filesystem::remove_all(temp_dir);
        return result;
      }

      std::ofstream out(fatbin_file, std::ios::binary);
      out.write(fatbin_data.data(), static_cast<std::streamsize>(fatbin_data.size()));
      if (!out)
      {
        result.diagnostics += "\nFailed to write fatbin file";
        std::filesystem::remove_all(temp_dir);
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
      std::filesystem::remove_all(temp_dir);
      return result;
    }

    std::filesystem::remove_all(temp_dir);
    result.success = true;
    return result;
  }

  LinkResult linkToSharedLibrary(
    const std::vector<std::string>& object_files, const std::string& output_path, const CompilerConfig& config)
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
        namespace fs = std::filesystem;
        if (!fs::exists(lib_path))
        {
          continue;
        }
        for (const auto& entry : fs::directory_iterator(lib_path))
        {
          auto fname = entry.path().filename().string();
          if (fname.starts_with("libcudart.so"))
          {
            arg_strings.push_back(entry.path().string());
            found_cudart = true;
            break;
          }
        }
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

CUDACompiler::CUDACompiler()
    : impl_(new Impl())
{}
CUDACompiler::~CUDACompiler()
{
  delete impl_;
}

BitcodeResult CUDACompiler::compileToDeviceBitcode(const std::string& source_code, const CompilerConfig& config)
{
  return impl_->compileToDeviceBitcode(source_code, config);
}

CompilationResult CUDACompiler::compileToObject(
  const std::string& source_code, const std::string& output_path, const CompilerConfig& config)
{
  return impl_->compileToObject(source_code, output_path, config);
}

LinkResult CUDACompiler::linkToSharedLibrary(
  const std::vector<std::string>& object_files, const std::string& output_path, const CompilerConfig& config)
{
  return impl_->linkToSharedLibrary(object_files, output_path, config);
}
} // namespace hostjit
