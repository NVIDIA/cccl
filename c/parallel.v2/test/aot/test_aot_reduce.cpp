// AoT (compile/load/serialize/deserialize) coverage for cccl_device_reduce
// on the v2 (HostJIT) backend.
//
// Covers, in order:
//   1. cccl_device_reduce_build_ex (fused) still works (no regression) and
//      leaves build->payload populated.
//   2. cccl_device_reduce_compile + cccl_device_reduce_load (explicit split,
//      same process) match the fused result.
//   3. cccl_device_reduce_serialize -> cccl_device_reduce_deserialize ->
//      cccl_device_reduce_load, same process.
//   4. The actual AoT claim: serialize in THIS process, write the blob to a
//      file, fork()+exec() a genuinely separate OS process (re-invoking this
//      same binary in "--child" mode) that ONLY calls deserialize+load+run
//      (no compile/build call at all) and exits 0 on success. The parent
//      asserts the child exited 0. This is a real cross-process load, not
//      just a fresh object in the same process/address space.
//   5. A blob with a corrupted os_arch_tag is rejected by deserialize with a
//      clear error, not a crash.
//
// Only the default well-known-op (LLVM-IR, inlined) path is exercised here.
// The CCCL_OP_LTOIR (nvJitLink, non-inlined) escape hatch is a separately
// selected op code path (see cccl_op_code_type in types.h) and is not
// covered by this test — it would need a custom op supplying real LTOIR,
// which needs NVRTC/nvcc tooling this freestanding test doesn't otherwise
// link against.

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <cccl/c/reduce.h>
#include <cccl/c/serialization.h>

#ifdef _WIN32
#  include <process.h>
#else
#  include <unistd.h>

#  include <sys/wait.h>
#endif

#define CUDA_CHECK(x)                                                                           \
  do                                                                                            \
  {                                                                                             \
    cudaError_t err = (x);                                                                      \
    if (err != cudaSuccess)                                                                     \
    {                                                                                           \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

namespace
{
constexpr int N = 1 << 14;

void get_cc(int& cc_major, int& cc_minor)
{
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  cc_major = prop.major;
  cc_minor = prop.minor;
}

// Runs a built/loaded reduce and returns the sum of N ones (expected: N).
// Fails the process (exit 1) on any CUDA or reduce error.
int run_reduce(cccl_device_reduce_build_result_t build)
{
  std::vector<int> h_in(N, 1);
  int *d_in, *d_out;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(int), cudaMemcpyHostToDevice));

  cccl_iterator_t d_in_it{};
  d_in_it.size             = sizeof(int);
  d_in_it.alignment        = alignof(int);
  d_in_it.type             = CCCL_POINTER;
  d_in_it.value_type       = cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
  d_in_it.state            = d_in;
  cccl_iterator_t d_out_it = d_in_it;
  d_out_it.state           = d_out;

  cccl_op_t op{};
  op.type = CCCL_PLUS;

  int h_init = 0;
  cccl_value_t init{};
  init.type  = cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
  init.state = &h_init;

  size_t temp_bytes = 0;
  CUresult rc       = cccl_device_reduce(build, nullptr, &temp_bytes, d_in_it, d_out_it, N, op, init, nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "cccl_device_reduce (temp-size query) failed: %d\n", (int) rc);
    exit(1);
  }
  void* d_temp = nullptr;
  if (temp_bytes)
  {
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
  }
  rc = cccl_device_reduce(build, d_temp, &temp_bytes, d_in_it, d_out_it, N, op, init, nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "cccl_device_reduce (execute) failed: %d\n", (int) rc);
    exit(1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  int h_out = 0;
  CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost));
  cudaFree(d_in);
  cudaFree(d_out);
  if (d_temp)
  {
    cudaFree(d_temp);
  }
  return h_out;
}

bool build_and_serialize(std::vector<char>& out_blob)
{
  int cc_major, cc_minor;
  get_cc(cc_major, cc_minor);

  cccl_iterator_t d_in{};
  d_in.size             = sizeof(int);
  d_in.alignment        = alignof(int);
  d_in.type             = CCCL_POINTER;
  d_in.value_type       = cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
  cccl_iterator_t d_out = d_in;

  cccl_op_t op{};
  op.type = CCCL_PLUS;

  int h_init = 0;
  cccl_value_t init{};
  init.type  = cccl_type_info{sizeof(int), alignof(int), CCCL_INT32};
  init.state = &h_init;

  const char* cub_path        = TEST_CUB_PATH;
  const char* thrust_path     = TEST_THRUST_PATH;
  const char* libcudacxx_path = TEST_LIBCUDACXX_PATH;

  // --- 1. build_ex (fused), confirm payload is populated ---
  cccl_device_reduce_build_result_t fused_build{};
  CUresult rc = cccl_device_reduce_build_ex(
    &fused_build,
    d_in,
    d_out,
    op,
    init,
    CCCL_NOT_GUARANTEED,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    nullptr,
    nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_build_ex: %d\n", (int) rc);
    return false;
  }
  if (fused_build.payload == nullptr || fused_build.payload_size == 0)
  {
    fprintf(stderr, "FAIL: build_ex did not populate payload\n");
    return false;
  }
  int fused_result = run_reduce(fused_build);
  if (fused_result != N)
  {
    fprintf(stderr, "FAIL: build_ex result=%d expected=%d\n", fused_result, N);
    return false;
  }
  printf("[1/5] build_ex (fused): OK, result=%d, payload=%zu bytes\n", fused_result, fused_build.payload_size);

  // --- 2. explicit compile + load, same process, compare to fused ---
  cccl_device_reduce_build_result_t split_build{};
  rc = cccl_device_reduce_compile(
    &split_build,
    d_in,
    d_out,
    op,
    init,
    CCCL_NOT_GUARANTEED,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    nullptr,
    nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_compile: %d\n", (int) rc);
    return false;
  }
  if (split_build.jit_compiler != nullptr || split_build.reduce_fn != nullptr)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_compile should not populate jit_compiler/reduce_fn\n");
    return false;
  }
  rc = cccl_device_reduce_load(&split_build, nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_load: %d\n", (int) rc);
    return false;
  }
  int split_result = run_reduce(split_build);
  if (split_result != N)
  {
    fprintf(stderr, "FAIL: compile+load result=%d expected=%d\n", split_result, N);
    return false;
  }
  printf("[2/5] compile()+load() (split, same process): OK, result=%d\n", split_result);

  // --- 3. serialize -> deserialize -> load, same process ---
  void* buf   = nullptr;
  size_t size = 0;
  rc          = cccl_device_reduce_serialize(&fused_build, &buf, &size);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_serialize: %d\n", (int) rc);
    return false;
  }
  printf("[3/5] serialize(): OK, blob=%zu bytes\n", size);

  cccl_device_reduce_build_result_t deser_build{};
  rc = cccl_device_reduce_deserialize(&deser_build, buf, size);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_deserialize: %d\n", (int) rc);
    return false;
  }
  if (deser_build.accumulator_size != fused_build.accumulator_size
      || deser_build.determinism != fused_build.determinism)
  {
    fprintf(stderr, "FAIL: deserialize did not preserve accumulator_size/determinism\n");
    return false;
  }
  rc = cccl_device_reduce_load(&deser_build, nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "FAIL: cccl_device_reduce_load (post-deserialize): %d\n", (int) rc);
    return false;
  }
  int deser_result = run_reduce(deser_build);
  if (deser_result != N)
  {
    fprintf(stderr, "FAIL: serialize+deserialize+load result=%d expected=%d\n", deser_result, N);
    return false;
  }
  printf("[3/5 cont'd] deserialize()+load() (same process): OK, result=%d\n", deser_result);

  // --- 5 (checked here since we already have a valid blob): corrupted
  // os_arch_tag is rejected cleanly, not a crash ---
  {
    std::vector<char> corrupted(static_cast<char*>(buf), static_cast<char*>(buf) + size);
    // os_arch_tag is the 3rd u32 in the header, right after magic[8] +
    // algo_tag(u32) -- see c/parallel.v2/src/util/serialization.h.
    uint32_t bogus_tag = 0xdeadbeef;
    std::memcpy(corrupted.data() + 8 + sizeof(uint32_t), &bogus_tag, sizeof(bogus_tag));
    cccl_device_reduce_build_result_t bogus_build{};
    CUresult bogus_rc = cccl_device_reduce_deserialize(&bogus_build, corrupted.data(), corrupted.size());
    if (bogus_rc == CUDA_SUCCESS)
    {
      fprintf(stderr, "FAIL: deserialize should have rejected a corrupted os_arch_tag\n");
      return false;
    }
    printf("[5/5] corrupted os_arch_tag correctly rejected (rc=%d)\n", (int) bogus_rc);
  }

  out_blob.assign(static_cast<char*>(buf), static_cast<char*>(buf) + size);
  cccl_serialization_v2_buffer_free(buf);
  return true;
}

// Child-process entry point: ONLY deserialize+load+run. No build/compile
// call at all -- this process never ran Clang, proving the blob is a
// genuinely self-contained, reloadable artifact.
int run_child(const char* blob_path)
{
  std::ifstream f(blob_path, std::ios::binary);
  if (!f)
  {
    fprintf(stderr, "child: could not open blob at %s\n", blob_path);
    return 1;
  }
  std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  f.close();

  cccl_device_reduce_build_result_t build{};
  CUresult rc = cccl_device_reduce_deserialize(&build, blob.data(), blob.size());
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "child: deserialize failed: %d\n", (int) rc);
    return 1;
  }
  rc = cccl_device_reduce_load(&build, nullptr);
  if (rc != CUDA_SUCCESS)
  {
    fprintf(stderr, "child: load failed: %d\n", (int) rc);
    return 1;
  }
  int result = run_reduce(build);
  printf("child: result=%d expected=%d -> %s\n", result, N, (result == N) ? "PASS" : "FAIL");
  return (result == N) ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
  if (argc == 3 && std::strcmp(argv[1], "--child") == 0)
  {
    return run_child(argv[2]);
  }
  if (argc != 1)
  {
    fprintf(stderr, "usage: %s (no args: run full test) | %s --child <blob-path>\n", argv[0], argv[0]);
    return 2;
  }

  std::vector<char> blob;
  if (!build_and_serialize(blob))
  {
    return 1;
  }

  char blob_path[] = "/tmp/cccl_v2_aot_test_XXXXXX";
#ifndef _WIN32
  int fd = mkstemp(blob_path);
  if (fd < 0)
  {
    fprintf(stderr, "FAIL: mkstemp failed\n");
    return 1;
  }
  ssize_t written = write(fd, blob.data(), blob.size());
  close(fd);
  if (written < 0 || static_cast<size_t>(written) != blob.size())
  {
    fprintf(stderr, "FAIL: could not write blob to temp file\n");
    return 1;
  }

  // --- 4. genuinely separate OS process: fork()+exec() this same binary in
  // --child mode. It NEVER calls compile/build -- only deserialize+load. ---
  pid_t pid = fork();
  if (pid < 0)
  {
    fprintf(stderr, "FAIL: fork() failed\n");
    unlink(blob_path);
    return 1;
  }
  if (pid == 0)
  {
    execl(argv[0], argv[0], "--child", blob_path, (char*) nullptr);
    // execl only returns on failure
    fprintf(stderr, "FAIL: execl failed: %s\n", std::strerror(errno));
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0)
  {
    fprintf(stderr, "FAIL: waitpid failed\n");
    unlink(blob_path);
    return 1;
  }
  unlink(blob_path);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
  {
    fprintf(stderr, "FAIL: child process (separate OS process, fresh dlopen) did not exit 0 (status=%d)\n", status);
    return 1;
  }
  printf("[4/5] cross-process deserialize()+load() (separate OS process, fork+exec): OK\n");
#else
  fprintf(stderr,
          "[4/5] SKIPPED on Windows: cross-process fork()+exec() coverage not yet implemented "
          "here; needs a CreateProcess-based equivalent.\n");
  (void) blob_path;
#endif

  printf("ALL AoT reduce checks PASSED\n");
  return 0;
}
