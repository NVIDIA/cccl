#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hostjit/jit_compiler.hpp>

#ifdef _WIN32
#  include <process.h>
#else
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

// 64-bit FNV-1a over everything that determines the contents of a PCH: the
// preamble text, the PCH kind, and every option handed to libnvcc.
//
// This is the cache key. Two compiles that agree on it can share a PCH; two
// that disagree must not, because clang validates a PCH against the command
// line it was built with. Keying on (kind, sm_version) alone -- as this did
// before -- means a source-tree build and an installed wheel, which differ in
// their include paths, collide on one entry and break each other.
//
// Header *contents* are deliberately not hashed: reading every CUB header to
// digest it would cost about as much as the parse the PCH exists to avoid. An
// in-place CCCL upgrade therefore keeps the same key and leaves a stale PCH
// behind; clang's own size/mtime validation rejects it, and the
// retry-without-PCH path in JITCompiler::compile recovers.
std::string hash_pch_options(const std::vector<std::string>& options, const std::string& kind_name)
{
  std::uint64_t h = 1469598103934665603ull;
  auto mix        = [&h](std::string_view s) {
    for (unsigned char c : s)
    {
      h ^= c;
      h *= 1099511628211ull;
    }
    h ^= '\0';
    h *= 1099511628211ull;
  };

  mix(pch_preamble_source);
  mix(kind_name);
  for (const auto& option : options)
  {
    mix(option);
  }

  // Fixed-width lowercase hex, so cache filenames sort and glob predictably.
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i)
  {
    out[static_cast<size_t>(i)] = "0123456789abcdef"[h & 0xf];
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
// by recency. Every use refreshes the timestamp: skipping recent ones would make
// an entry rebuilt and then hammered look older than one touched once and
// abandoned, which inverts the ordering exactly where it matters. One timestamp
// write is immaterial next to the compile it accompanies.
//
// Only the .pch is touched. A PCH records its preamble as an input and clang
// validates that input's mtime, so rewriting the preamble's timestamp would
// invalidate the very entry this is trying to keep alive. The preamble needs no
// timestamp of its own: it is evicted with its .pch, never independently.
void touch_pch_entry(const std::string& pch_path)
{
  const auto now = std::filesystem::file_time_type::clock::now();
  std::error_code ec;
  std::filesystem::last_write_time(pch_path, now, ec);
}

// Best-effort exclusive guard around PCH generation, held across processes and
// threads alike.
//
// Generating a PCH costs seconds of CPU and writes tens of megabytes. Without a
// guard, every process that starts against a cold cache generates its own copy
// of the same file -- a test runner fanning out eight workers against a cold
// cache would otherwise do eight identical generations at once.
//
// Deliberately non-blocking: a caller that cannot take the lock builds without
// a PCH instead of waiting for the holder. Waiting would trade one slow build
// for a slow build *plus* a stall, and would need timeout handling for a holder
// that died. Taking the slow path costs exactly what the build would have cost
// with PCH disabled.
//
// A directory is the lock: create_directory is atomic on POSIX and Windows and
// reports whether it did the creating, which is precisely test-and-set.
//
// A lock left behind by a killed process is reclaimed by whoever prunes the
// cache, not here. How long to wait before presuming a holder dead is a
// judgment about the cache rather than a property of taking a lock, and it
// belongs with the rest of the pruning rules.
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
  std::string& pch_path,
  bool& generated)
{
  pch_path.clear();

  // The caller supplies the location; this library never picks one. An empty
  // path means "no PCH", which is how a caller that does not want the feature
  // (or could not resolve a writable directory) turns it off.
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

  std::vector<std::string> options;
  config.appendCommandLineArguments(options);

  // The arch appears in the filename for readability only -- the hash already
  // covers it, because libnvcc takes one option list for both PCH kinds and
  // that list carries the target. (Clang's *host* frontend arguments contain no
  // arch, so host PCHs could in principle be shared across GPUs; that is not
  // exploitable here without assuming what libnvcc does with the option
  // internally, and guessing wrong would mean handing a compile the wrong PCH.)
  const std::string label = kind_name + "_sm" + std::to_string(config.sm_version);
  const std::string key   = hash_pch_options(options, kind_name);
  pch_path                = get_pch_path(cache_dir, label, key);
  const auto source_path  = get_pch_source_path(cache_dir, label, key);

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

  auto pch_result = libnvccCreatePCH(
    program.program,
    kind,
    source_path.c_str(),
    pch_path.c_str(),
    static_cast<int>(option_ptrs.size()),
    option_ptrs.empty() ? nullptr : option_ptrs.data());
  if (pch_result != LIBNVCC_SUCCESS)
  {
    diagnostics += kind_name + " PCH generation failed: " + hostjit::detail::get_libnvcc_program_log(program.program);
    diagnostics += "\n";
    pch_path.clear();
    return false;
  }

  generated = true;
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

  bool generated = false;

  std::string device_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_DEVICE, "device", diagnostics, device_pch_path, generated))
  {
    prepared.device_pch_path = std::move(device_pch_path);
  }

  std::string host_pch_path;
  if (create_pch_if_needed(prepared, LIBNVCC_PCH_HOST, "host", diagnostics, host_pch_path, generated))
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

bool JITCompiler::compile(const std::string& source_code)
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
    discard_pch(libnvcc_config.device_pch_path);
    discard_pch(libnvcc_config.host_pch_path);
    if (config_.verbose)
    {
      std::cout << "PCH rejected by the compiler; retrying without it\n";
    }

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

  for (int attempt = 0; attempt < 10; ++attempt)
  {
    std::string dir_name            = "hostjit_" + std::to_string(pid) + "_" + std::to_string(dis(gen));
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
