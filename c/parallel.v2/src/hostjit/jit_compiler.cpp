#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <process.h>
#  include <windows.h>
#else
#  include <dlfcn.h>
#  include <unistd.h>
#endif

namespace
{
static constexpr const char* pch_preamble_source =
  "#include <cuda_runtime.h>\n"
  "#include <cuda/std/iterator>\n"
  "#include <cuda/std/functional>\n"
  "#include <cuda/functional>\n"
  "#include <cub/device/device_adjacent_difference.cuh>\n"
  "#include <cub/device/device_copy.cuh>\n"
  "#include <cub/device/device_find.cuh>\n"
  "#include <cub/device/device_for.cuh>\n"
  "#include <cub/device/device_histogram.cuh>\n"
  "#include <cub/device/device_merge.cuh>\n"
  "#include <cub/device/device_merge_sort.cuh>\n"
  "#include <cub/device/device_partition.cuh>\n"
  "#include <cub/device/device_radix_sort.cuh>\n"
  "#include <cub/device/device_reduce.cuh>\n"
  "#include <cub/device/device_scan.cuh>\n"
  "#include <cub/device/device_segmented_radix_sort.cuh>\n"
  "#include <cub/device/device_segmented_scan.cuh>\n"
  "#include <cub/device/device_segmented_sort.cuh>\n"
  "#include <cub/device/device_select.cuh>\n"
  "#include <cub/device/device_transform.cuh>\n";

// The cache key is everything that determines the contents of a PCH: the preamble text,
// the PCH kind, and every option handed to libnvcc.
std::string hash_pch_options(const std::vector<std::string>& options, const std::string& kind_name)
{
  std::string blob;
  const auto append = [&blob](std::string_view s) {
    blob.append(s);
    blob.push_back('\0');
  };

  append(pch_preamble_source);
  append(kind_name);
  if (kind_name == "host")
  {
    // Bump when the host PCH's hardcoded -include list changes (see
    // LIBNVCC_PCH_HOST in generatePCH): that list isn't part of `options`
    // (config.appendCommandLineArguments only serializes CompilerConfig
    // fields), so without this a stale cached PCH from before such a change
    // would otherwise be reused as-is.
    append("host-pch-cxa-guard-v1");
  }
  for (const auto& option : options)
  {
    append(option);
  }

  std::size_t h = std::hash<std::string_view>{}(blob);

  // Fixed-width lowercase hex, so cache filenames sort and glob predictably.
  std::string out(2 * sizeof(h), '0');
  for (auto i = out.size(); i-- > 0;)
  {
    out[i] = "0123456789abcdef"[h & 0xf];
    h >>= 4;
  }
  return out;
}

std::string get_pch_path(const std::filesystem::path& dir, const std::string& kind, const std::string& key)
{
  return (dir / (kind + "_" + key + ".pch")).string();
}

std::string get_pch_source_path(const std::filesystem::path& dir, const std::string& kind, const std::string& key)
{
  return (dir / (kind + "_" + key + "_preamble.cu")).string();
}

// Record that an entry was used, so whoever prunes the cache can order entries
// by recency.
void touch_pch_entry(const std::string& pch_path)
{
  const auto now = std::filesystem::file_time_type::clock::now();
  std::error_code ec;
  std::filesystem::last_write_time(pch_path, now, ec);
}

// Best-effort exclusive guard around PCH generation, held across processes and
// threads alike. Deliberately non-blocking: a caller that cannot take the lock
// builds without a PCH instead of waiting for the holder.
//
// A directory is the lock: create_directory is atomic on POSIX and Windows and
// reports whether it did the creating, which is precisely test-and-set.
class PCHGenerationLock
{
public:
  explicit PCHGenerationLock(std::filesystem::path lock_path)
      : path_(std::move(lock_path))
  {
    std::error_code ec;
    held_ = std::filesystem::create_directory(path_, ec) && !ec;
  }

  ~PCHGenerationLock()
  {
    if (held_)
    {
      std::error_code ec;
      std::filesystem::remove(path_, ec);
    }
  }

  PCHGenerationLock(const PCHGenerationLock&)            = delete;
  PCHGenerationLock& operator=(const PCHGenerationLock&) = delete;

  bool held() const
  {
    return held_;
  }

private:
  std::filesystem::path path_;
  bool held_ = false;
};

bool create_pch_if_needed(
  hostjit::CompilerConfig config,
  libnvccPCHKind kind,
  const std::string& kind_name,
  std::string& diagnostics,
  std::string& pch_path)
{
  pch_path.clear();

  const std::filesystem::path cache_dir{config.pch_cache_dir};
  if (cache_dir.empty())
  {
    diagnostics += "no PCH cache directory supplied; building without PCH\n";
    return false;
  }

  std::error_code mkdir_ec;
  std::filesystem::create_directories(cache_dir, mkdir_ec);
  std::error_code is_dir_ec;
  if (mkdir_ec && !std::filesystem::is_directory(cache_dir, is_dir_ec))
  {
    diagnostics += "PCH cache directory is unusable; building without PCH\n";
    return false;
  }

  // Generate against the same options a later compile will use, minus the PCH
  // settings themselves -- those are what we are producing.
  config.enable_pch = false;
  config.device_pch_path.clear();
  config.host_pch_path.clear();

  // Strip everything that varies per build. None of it can affect a PCH: the
  // preamble is headers only, the user's operator is linked as bitcode after
  // the frontend runs, and the entry point name only drives post-compile
  // passes. Leaving them in would put `--device-bitcode=<temp path>` and
  // `--entry-point=<algo>` into the cache key, so every build would generate
  // its own PCH -- tens of them per test run, none ever reused.
  config.device_bitcode_files.clear();
  config.device_ltoir_files.clear();
  config.entry_point_name.clear();

  // Diagnostic-only flags (--verbose, --trace-includes, --keep-artifacts) don't
  // change the PCH's contents either, so drop them from the key -- otherwise the
  // same headers built with and without them would fragment into two entries.
  config.verbose        = false;
  config.trace_includes = false;
  config.keep_artifacts = false;

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);

  const std::string label = kind_name + "_sm" + std::to_string(config.sm_version);
  // The PTX ISA version is derived inside libnvcc (from sm_version and the nvJitLink in use) rather
  // than serialized into `options`, but it is baked into a device PCH as a target feature, so it
  // has to be part of the cache key. Hash a copy: `options` is what libnvcc is invoked with.
  std::vector<std::string> key_options = options;
  if (kind_name == "device")
  {
    key_options.push_back("ptx-isa=" + std::to_string(libnvccPtxVersionForSm(config.sm_version)));
  }
  const std::string key  = hash_pch_options(key_options, kind_name);
  pch_path               = get_pch_path(cache_dir, label, key);
  const auto source_path = get_pch_source_path(cache_dir, label, key);

  auto present = [&] {
    std::error_code ec;
    return std::filesystem::exists(pch_path, ec) && !ec;
  };

  if (present())
  {
    touch_pch_entry(pch_path);
    return true;
  }

  // Only one generator at a time; losers build without a PCH rather than
  // waiting. See PCHGenerationLock.
  PCHGenerationLock lock(cache_dir / (label + "_" + key + ".lock"));
  if (!lock.held())
  {
    // The holder may have finished between the check above and here.
    if (present())
    {
      touch_pch_entry(pch_path);
      return true;
    }
    diagnostics += kind_name + " PCH generation already in progress elsewhere; building without PCH\n";
    pch_path.clear();
    return false;
  }

  // Re-check under the lock: another generator may have landed the file while
  // we were acquiring it.
  if (present())
  {
    touch_pch_entry(pch_path);
    return true;
  }

  auto option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

  hostjit::detail::LibnvccProgramGuard program;
  auto create_result = libnvccCreateProgram(&program.program, pch_preamble_source, "hostjit_preamble.cu");
  if (create_result != LIBNVCC_SUCCESS)
  {
    diagnostics += "Failed to create libnvcc PCH program: ";
    diagnostics += libnvccGetErrorString(create_result);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }

  // Generate to a temp path and rename into place: libnvccCreatePCH does not
  // publish atomically, so a concurrent build could read a half-written
  // pch_path. The temp is unique per writer (pid + random) because the
  // generation lock is only best-effort, so two writers can still race here;
  // the cache sweep reclaims temps orphaned by a killed writer.
#ifdef _WIN32
  const long long pch_tmp_pid = _getpid();
#else
  const long long pch_tmp_pid = ::getpid();
#endif
  static thread_local std::mt19937_64 pch_tmp_rng{std::random_device{}()};
  const std::string tmp_path =
    pch_path + "." + std::to_string(pch_tmp_pid) + "." + std::to_string(pch_tmp_rng()) + ".tmp";
  std::error_code tmp_ec;

  auto pch_result = libnvccCreatePCH(
    program.program,
    kind,
    source_path.c_str(),
    tmp_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  if (pch_result != LIBNVCC_SUCCESS)
  {
    diagnostics += kind_name + " PCH generation failed: " + hostjit::detail::get_libnvcc_program_log(program.program);
    diagnostics += "\n";
    std::filesystem::remove(tmp_path, tmp_ec);
    pch_path.clear();
    return false;
  }

  std::error_code rename_ec;
  std::filesystem::rename(tmp_path, pch_path, rename_ec);
  if (rename_ec)
  {
    diagnostics += kind_name + " PCH publish failed: " + rename_ec.message() + "\n";
    std::filesystem::remove(tmp_path, rename_ec);
    pch_path.clear();
    return false;
  }

  return true;
}

// Discard a PCH the compiler rejected, so the next build regenerates it instead
// of tripping over the same entry forever. Best-effort by design: another
// process may be reading or replacing it concurrently.
void discard_pch(const std::string& pch_path)
{
  if (pch_path.empty())
  {
    return;
  }
  std::error_code ec;
  std::filesystem::remove(pch_path, ec);
}

hostjit::CompilerConfig prepare_pch_config(const hostjit::CompilerConfig& config, std::string& diagnostics)
{
  hostjit::CompilerConfig prepared = config;
  prepared.device_pch_path.clear();
  prepared.host_pch_path.clear();

  if (!prepared.enable_pch)
  {
    return prepared;
  }

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_DEVICE, "device", diagnostics, device_pch_path))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_HOST, "host", diagnostics, host_pch_path))
  {
    prepared.host_pch_path = std::move(host_pch_path);
  }

  return prepared;
}

bool read_file(const std::string& path, std::vector<char>& out)
{
  std::ifstream f(path, std::ios::binary);
  if (!f)
  {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
  return true;
}
} // anonymous namespace

namespace hostjit
{
JITCompiler::JITCompiler()
    : config_(detectDefaultConfig())
{}

JITCompiler::JITCompiler(const CompilerConfig& config)
    : config_(config)
{}

JITCompiler::~JITCompiler()
{
  cleanup();
}

// Shared by compile() and compileOnly(): validate config, compile to an
// object file, link to a shared library in a fresh temp dir. Does NOT load
// it. On success, *out_lib_path is the path to the linked artifact (still
// on disk in temp_dir_, which the caller may read before it's torn down by
// a later cleanup()).
bool JITCompiler::compileAndLink(const std::string& source_code, std::string* out_lib_path)
{
  std::string config_error;
  if (!validateConfig(config_, &config_error))
  {
    last_error_ = "Configuration error: " + config_error;
    return false;
  }

  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

  std::string pch_diagnostics;
  CompilerConfig libnvcc_config = prepare_pch_config(config_, pch_diagnostics);
  if (config_.verbose && !pch_diagnostics.empty())
  {
    std::cout << pch_diagnostics;
  }

  std::vector<std::string> options;
  libnvcc_config.appendCommandLineArguments(options);
  auto option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

  hostjit::detail::LibnvccProgramGuard program;
  auto create_result = libnvccCreateProgram(&program.program, source_code.c_str(), "input.cu");
  if (create_result != LIBNVCC_SUCCESS)
  {
    last_error_ = std::string("Failed to create libnvcc program: ") + libnvccGetErrorString(create_result);
    removeTempDirectory();
    return false;
  }

  std::string obj_path   = temp_dir_ + "/cuda_code.o";
  std::string cubin_path = temp_dir_ + "/device.cubin";
  auto compile_result    = libnvccCompileProgramToObject(
    program.program,
    obj_path.c_str(),
    cubin_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto compile_log = hostjit::detail::get_libnvcc_program_log(program.program);

  // A PCH the compiler rejects -- stale after an in-place header upgrade, or
  // written by a build whose options happened to hash the same -- must not be
  // able to fail the build. Drop it and compile the honest way. Clang validates
  // a PCH against both the command line and the size/mtime of every header it
  // covers, so a rejected PCH is a routine possibility rather than a defect.
  const bool used_pch = !libnvcc_config.device_pch_path.empty() || !libnvcc_config.host_pch_path.empty();
  if (compile_result != LIBNVCC_SUCCESS && used_pch)
  {
    if (config_.verbose)
    {
      std::cout << "Compile failed with a PCH; retrying without it\n";
    }

    // Keep the paths so the PCHs are discarded only if the retry succeeds.
    const std::string rejected_device_pch = libnvcc_config.device_pch_path;
    const std::string rejected_host_pch   = libnvcc_config.host_pch_path;

    libnvcc_config.enable_pch = false;
    libnvcc_config.device_pch_path.clear();
    libnvcc_config.host_pch_path.clear();

    options.clear();
    libnvcc_config.appendCommandLineArguments(options);
    option_ptrs = hostjit::detail::make_libnvcc_option_ptrs(options);

    // A fresh program: the failed attempt's log would otherwise be reported
    // alongside the retry's, so a genuine compile error would appear twice.
    hostjit::detail::LibnvccProgramGuard retry_program;
    auto retry_create = libnvccCreateProgram(&retry_program.program, source_code.c_str(), "input.cu");
    if (retry_create != LIBNVCC_SUCCESS)
    {
      last_error_ = std::string("Failed to create libnvcc program: ") + libnvccGetErrorString(retry_create);
      removeTempDirectory();
      return false;
    }

    compile_result = libnvccCompileProgramToObject(
      retry_program.program,
      obj_path.c_str(),
      cubin_path.c_str(),
      static_cast<int>(option_ptrs.size()),
      option_ptrs.empty() ? nullptr : option_ptrs.data());
    compile_log = hostjit::detail::get_libnvcc_program_log(retry_program.program);

    // The retry owns the rest of this build: linking below reuses `program`'s
    // handle for its log, so hand ownership over.
    std::swap(program.program, retry_program.program);

    // Discard the PCHs only if building without them succeeded -- that proves
    // they were at fault. If the retry also failed, the error is in the user's
    // program, so keep the (valid) PCHs.
    if (compile_result == LIBNVCC_SUCCESS)
    {
      discard_pch(rejected_device_pch);
      discard_pch(rejected_host_pch);
    }
  }

  if (compile_result != LIBNVCC_SUCCESS)
  {
    last_error_ = "Compilation failed:\n" + compile_log;
    removeTempDirectory();
    return false;
  }

  cubin_.clear();
  if (!read_file(cubin_path, cubin_))
  {
    last_error_ = "Compilation failed: generated cubin could not be read";
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Compilation diagnostics:\n" << compile_log << "\n";
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif
  const char* object_files[] = {obj_path.c_str()};
  auto link_result           = libnvccLinkToSharedLibrary(
    program.program,
    1,
    object_files,
    lib_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  auto link_log = hostjit::detail::get_libnvcc_program_log(program.program);

  if (link_result != LIBNVCC_SUCCESS)
  {
    last_error_ = "Linking failed:\n" + link_log;
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Linking diagnostics:\n" << link_log << "\n";
  }

  *out_lib_path = lib_path;
  return true;
}

bool JITCompiler::compile(const std::string& source_code)
{
  std::string lib_path;
  if (!compileAndLink(source_code, &lib_path))
  {
    return false;
  }

  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  if (config_.verbose)
  {
    std::cout << "Successfully loaded library: " << lib_path << "\n";
  }

  last_error_.clear();
  return true;
}

bool JITCompiler::compileOnly(const std::string& source_code)
{
  std::string lib_path;
  if (!compileAndLink(source_code, &lib_path))
  {
    return false;
  }
  // Deliberately do NOT dlopen/LoadLibrary — the caller only wants the
  // artifact bytes (via getArtifactsPath()), e.g. to persist them for AoT.
  // temp_dir_ (and the artifact within it) stays on disk until the next
  // compile()/compileOnly()/loadFromBytes() call or destruction — the
  // caller must read the bytes before then.
  last_error_.clear();
  return true;
}

namespace
{
// CUDART_VERSION encodes a release as major * 1000 + minor * 10.
std::string cudartVersionString(int version)
{
  return std::to_string(version / 1000) + "." + std::to_string((version % 1000) / 10);
}

// An artifact imports cudart symbols from the headers it was built against:
// the runtime must have the same major (the soname changes) and a minor at
// least as new (a minor only adds exports). Returns an error message, or an
// empty string if `local_version` is acceptable. A version of 0 is unknown.
std::string checkCudartCompatible(int built_version, int local_version)
{
  if (built_version == 0)
  {
    return "the CUDA runtime version the artifact was built against is unknown";
  }
  if (local_version == 0)
  {
    return "cannot determine its version (cudaRuntimeGetVersion failed)";
  }
  if (built_version / 1000 == local_version / 1000 && local_version >= built_version)
  {
    return {};
  }
  return "the artifact was built against CUDA runtime " + cudartVersionString(built_version) + ", but this machine has "
       + cudartVersionString(local_version)
       + "; the runtime must have the same major version and a minor at least as new as the build's "
         "(build ahead-of-time artifacts with the oldest CUDA Toolkit you need to support)";
}
} // namespace

bool JITCompiler::loadFromBytes(
  const std::vector<char>& library_bytes, const std::string& cudart_path, int built_cudart_version)
{
  // Deliberately no validateConfig(): that requires the toolkit's headers, but
  // loading a prebuilt artifact only needs the runtime library, which a
  // load-only machine may well have without the headers.
  cleanup();

  temp_dir_ = createTempDirectory();
  if (temp_dir_.empty())
  {
    last_error_ = "Failed to create temporary directory";
    return false;
  }

#ifdef _WIN32
  std::string lib_path = temp_dir_ + "/cuda_code.dll";
#else
  std::string lib_path = temp_dir_ + "/libcuda_code.so";
#endif

  {
    std::ofstream out(lib_path, std::ios::binary);
    if (!out)
    {
      last_error_ = "Failed to open " + lib_path + " for writing";
      removeTempDirectory();
      return false;
    }
    out.write(library_bytes.data(), static_cast<std::streamsize>(library_bytes.size()));
    if (!out)
    {
      last_error_ = "Failed to write artifact bytes to " + lib_path;
      removeTempDirectory();
      return false;
    }
  }

  // Cross-machine portability: load this machine's own CUDA runtime *before*
  // the artifact, so the artifact's cudart dependency resolves to it.
  // - Linux: the artifact's baked RPATH points at the machine it was BUILT on,
  //   which may not have CUDA at that path. glibc checks already-mapped
  //   objects by SONAME before any path search, so a preloaded
  //   libcudart.so.<major> (RTLD_GLOBAL) satisfies the dependency.
  // - Windows: the artifact names cudart64_<major>.dll; the loader reuses an
  //   already-loaded module of that name before searching, so the dependency
  //   no longer relies on the toolkit's bin directory being on PATH.
  // Never unloaded (matches the "never dlclose a JIT module" policy in
  // loader.cpp, see #9367: other loaded JIT modules may still reference it).
  {
    using get_version_fn = int (*)(int*);
#ifdef _WIN32
    SetLastError(0);
    HMODULE cudart_handle =
      LoadLibraryExW(std::filesystem::path(cudart_path).wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!cudart_handle)
    {
      last_error_ = "Failed to preload CUDA runtime (" + cudart_path + "): error " + std::to_string(GetLastError());
      removeTempDirectory();
      return false;
    }
    auto get_version = reinterpret_cast<get_version_fn>(GetProcAddress(cudart_handle, "cudaRuntimeGetVersion"));
#else
    dlerror();
    void* cudart_handle = dlopen(cudart_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!cudart_handle)
    {
      const char* err = dlerror();
      last_error_     = "Failed to preload CUDA runtime (" + cudart_path + "): " + (err ? err : "unknown dlopen error");
      removeTempDirectory();
      return false;
    }
    auto get_version = reinterpret_cast<get_version_fn>(dlsym(cudart_handle, "cudaRuntimeGetVersion"));
#endif
    // Fail early, with the cause, instead of with a raw loader error or an
    // undefined-symbol error from loading the artifact.
    int local_version = 0;
    if (get_version)
    {
      int version = 0;
      if (get_version(&version) == 0)
      {
        local_version = version;
      }
    }
    if (auto error = checkCudartCompatible(built_cudart_version, local_version); !error.empty())
    {
      last_error_ = "Incompatible CUDA runtime " + cudart_path + ": " + error;
      removeTempDirectory();
      return false;
    }
  }
  if (!library_.load(lib_path))
  {
    last_error_ = "Failed to load library: " + library_.getLastError();
    removeTempDirectory();
    return false;
  }

  last_error_.clear();
  return true;
}

void JITCompiler::cleanup()
{
  library_.unload();

  if (!config_.keep_artifacts)
  {
    removeTempDirectory();
  }

  last_error_.clear();
}

std::string JITCompiler::createTempDirectory()
{
  std::filesystem::path base_tmp_dir;

#ifdef _WIN32
  const char* tmp_dir = std::getenv("TEMP");
  if (!tmp_dir)
  {
    tmp_dir = std::getenv("TMP");
  }
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = std::filesystem::temp_directory_path();
  }
#else
  const char* tmp_dir = std::getenv("TMPDIR");
  if (tmp_dir)
  {
    base_tmp_dir = tmp_dir;
  }
  else
  {
    base_tmp_dir = "/tmp";
  }
#endif

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999999);

#ifdef _WIN32
  int pid = _getpid();
#else
  int pid = getpid();
#endif

  // JIT modules are never dlclose'd (see DynamicLibrary::unload), and dlopen returns the existing
  // handle when asked for a path that is already loaded -- even if the file at that path has since
  // been deleted and recreated. A directory name that repeats within this process would therefore
  // make a later build silently bind to an earlier module and fail to find its entry point. The
  // random suffix alone is only 20 bits, which collides in practice once a process has compiled a
  // few hundred modules, so also include a process-wide counter to make every name unique.
  static std::atomic<std::uint64_t> next_id{0};

  for (int attempt = 0; attempt < 10; ++attempt)
  {
    std::string dir_name =
      "hostjit_" + std::to_string(pid) + "_" + std::to_string(next_id.fetch_add(1)) + "_" + std::to_string(dis(gen));
    std::filesystem::path full_path = base_tmp_dir / dir_name;

    std::error_code ec;
    if (std::filesystem::create_directories(full_path, ec) && !ec)
    {
      return full_path.string();
    }
  }

  return "";
}

void JITCompiler::removeTempDirectory()
{
  if (temp_dir_.empty())
  {
    return;
  }

  try
  {
    if (std::filesystem::exists(temp_dir_))
    {
      std::filesystem::remove_all(temp_dir_);
    }
  }
  catch (const std::filesystem::filesystem_error& e)
  {
    if (config_.verbose)
    {
      std::cerr << "Warning: Failed to remove temporary directory: " << e.what() << "\n";
    }
  }

  temp_dir_.clear();
}
} // namespace hostjit
