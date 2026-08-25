#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Result codes returned by the libnvcc C API.
 *
 * Every libnvcc API function returns one of these values. Use
 * libnvccGetErrorString to obtain a stable string for logging or diagnostics.
 * Detailed compiler, linker, and tool diagnostics are stored on the program
 * handle and can be queried with libnvccGetProgramLogSize and
 * libnvccGetProgramLog.
 */
typedef enum libnvccResult
{
  LIBNVCC_SUCCESS = 0,
  LIBNVCC_ERROR_OUT_OF_MEMORY,
  LIBNVCC_ERROR_PROGRAM_CREATION_FAILURE,
  LIBNVCC_ERROR_INVALID_INPUT,
  LIBNVCC_ERROR_INVALID_PROGRAM,
  LIBNVCC_ERROR_INVALID_OPTION,
  LIBNVCC_ERROR_COMPILATION,
  LIBNVCC_ERROR_LINKING,
  LIBNVCC_ERROR_PCH_CREATE,
  LIBNVCC_ERROR_INTERNAL_ERROR
} libnvccResult;

/**
 * \brief Selects which Clang compilation mode is used to create a PCH.
 *
 * `LIBNVCC_PCH_DEVICE` creates a device-side PCH that can later be supplied to
 * libnvccCompileProgramToObject or libnvccCompileProgramToDeviceBitcode with
 * `--device-pch=<path>`. `LIBNVCC_PCH_HOST` creates a host-side PCH that can
 * later be supplied to libnvccCompileProgramToObject with `--host-pch=<path>`.
 */
typedef enum libnvccPCHKind
{
  LIBNVCC_PCH_DEVICE = 0,
  LIBNVCC_PCH_HOST   = 1
} libnvccPCHKind;

/**
 * \brief Opaque libnvcc program handle.
 *
 * A program owns the CUDA source string supplied to libnvccCreateProgram and
 * stores diagnostics from the most recent libnvcc operation involving that
 * program. Destroy it with libnvccDestroyProgram when no further compilation,
 * PCH creation, linking, or log retrieval is required.
 */
typedef struct libnvccProgram_st* libnvccProgram;

/**
 * \brief Return a static string describing a libnvcc result code.
 *
 * The returned pointer is owned by libnvcc and remains valid for the lifetime
 * of the process. Unknown result codes return `"LIBNVCC_ERROR_UNKNOWN"`.
 */
const char* libnvccGetErrorString(libnvccResult result);

/**
 * \brief Create a libnvcc program from a CUDA source string.
 *
 * \param prog Output location for the new program handle.
 * \param src NUL-terminated CUDA C++ source string. libnvcc copies this string.
 * \param name Optional logical source name used in diagnostics. When NULL or
 * empty, libnvcc uses `"input.cu"`.
 *
 * The program does not perform compilation during creation. Compile, link, and
 * PCH functions accept command-line options independently, similar to NVRTC.
 */
libnvccResult libnvccCreateProgram(libnvccProgram* prog, const char* src, const char* name);

/**
 * \brief Destroy a libnvcc program.
 *
 * \param prog Address of a program handle previously returned by
 * libnvccCreateProgram. On success, `*prog` is set to NULL. Passing NULL or a
 * pointer to NULL is accepted and returns LIBNVCC_SUCCESS.
 */
libnvccResult libnvccDestroyProgram(libnvccProgram* prog);

/**
 * \brief Compile a program's device source to LLVM bitcode and write it to a file.
 *
 * \param prog Program handle created by libnvccCreateProgram.
 * \param outputBitcodePath Destination path for the generated LLVM bitcode.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. The array may be NULL
 * when `numOptions` is zero.
 *
 * Supported options:
 * `--cuda-path=<path>`, `--hostjit-include-path=<path>`,
 * `--clang-headers-path=<path>`, `-isystem <path>`, `-isystem<path>`,
 * `--system-include-path=<path>`, `-I<path>`, `--include-path=<path>`, `-L<path>`,
 * `--library-path=<path>`, `--device-bitcode=<path>`,
 * `--device-ltoir=<path>`, `-D<name>[=<value>]`,
 * `--define-macro=<name>[=<value>]`, `--gpu-architecture=sm_<NN>`,
 * `--gpu-architecture=<NN>`, `-O<N>`, `--optimization-level=<N>`,
 * `--debug`, `--verbose`, `--trace-includes`, `--keep-artifacts`,
 * `--entry-point=<name>`, `--device-pch=<path>`, `--host-pch=<path>`,
 * `-XClang <arg>`, and `-XClang=<arg>`.
 *
 * This function uses only file paths for extra LLVM inputs. Source code is the
 * only in-memory input accepted by libnvcc.
 */
libnvccResult libnvccCompileProgramToDeviceBitcode(
  libnvccProgram prog, const char* outputBitcodePath, int numOptions, const char* const* options);

/**
 * \brief Compile a program to a host object file and optionally a cubin file.
 *
 * \param prog Program handle created by libnvccCreateProgram.
 * \param outputObjectPath Destination path for the generated host object file.
 * \param outputCubinPath Optional destination path for the linked device cubin.
 * Pass NULL or an empty string when the cubin is not needed.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. The array may be NULL
 * when `numOptions` is zero.
 *
 * Device LLVM bitcode and LTOIR inputs must be supplied with
 * `--device-bitcode=<path>` and `--device-ltoir=<path>`. PCH files are used
 * only when explicit `--device-pch=<path>` or `--host-pch=<path>` options are
 * present; libnvcc does not create or cache them implicitly.
 */
libnvccResult libnvccCompileProgramToObject(
  libnvccProgram prog,
  const char* outputObjectPath,
  const char* outputCubinPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Link object files into a shared library.
 *
 * \param prog Program handle used to store diagnostics from the link step.
 * \param numObjectFiles Number of entries in `objectFiles`.
 * \param objectFiles Array of object file paths to link.
 * \param outputLibraryPath Destination path for the linked shared library.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. Link-time options use
 * the same option parser as compile-time options; currently `--cuda-path`,
 * `-L`, `--library-path`, and `--verbose` affect linking.
 */
libnvccResult libnvccLinkToSharedLibrary(
  libnvccProgram prog,
  int numObjectFiles,
  const char* const* objectFiles,
  const char* outputLibraryPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Create a Clang PCH file for a program.
 *
 * \param prog Program handle whose source string is used as the PCH input.
 * \param kind Selects device or host compilation mode.
 * \param pchSourcePath Stable source path to write before invoking Clang.
 * Clang records this path in the PCH, so callers should use a cache-stable
 * location rather than a per-build temporary path.
 * \param pchOutputPath Destination path for the generated PCH file.
 * \param numOptions Number of entries in `options`.
 * \param options Array of command-line option strings. PCH creation uses the
 * same include, macro, architecture, optimization, and `-XClang` options as
 * compilation.
 *
 * libnvcc creates exactly the requested PCH file. It does not decide cache
 * locations, check freshness, or enable PCH use for later compilations.
 */
libnvccResult libnvccCreatePCH(
  libnvccProgram prog,
  libnvccPCHKind kind,
  const char* pchSourcePath,
  const char* pchOutputPath,
  int numOptions,
  const char* const* options);

/**
 * \brief Get the byte size of the program diagnostic log.
 *
 * The returned size includes the trailing NUL byte. Warnings and informational
 * messages may be present even when the preceding operation returned
 * LIBNVCC_SUCCESS.
 */
libnvccResult libnvccGetProgramLogSize(libnvccProgram prog, size_t* logSizeRet);

/**
 * \brief Copy the program diagnostic log into caller-provided storage.
 *
 * The caller must allocate at least the number of bytes returned by
 * libnvccGetProgramLogSize. The copied log is NUL-terminated.
 */
libnvccResult libnvccGetProgramLog(libnvccProgram prog, char* log);

#ifdef __cplusplus
}
#endif
