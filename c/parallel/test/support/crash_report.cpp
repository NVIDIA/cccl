//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! Prints the faulting address and a symbolized stack when a C Parallel test
//! dies from a native fault.
//!
//! Catch2's fatal-signal handler can only report the last *completed*
//! assertion, so a crash inside a long-running call shows up as
//!
//!     test_util.h(706): FAILED:
//!       {Unknown expression after the reported line}
//!     due to a fatal error condition:
//!       SIGSEGV - Segmentation violation signal
//!
//! which does not say where the process actually faulted. See NVIDIA/cccl#10802.
//!
//! This installs a vectored exception handler that runs before Catch2's filter,
//! prints the crash location, and then returns EXCEPTION_CONTINUE_SEARCH so that
//! Catch2 and ctest report exactly as they do today. The output is purely
//! additive.

// _WIN32 is also defined for Windows on ARM64, where CONTEXT exposes Pc/Sp/Fp
// instead of Rip/Rsp/Rbp and StackWalk64 needs IMAGE_FILE_MACHINE_ARM64. The
// supported Windows configuration is x64, so restrict this to x64 rather than
// carry an untested second stack walker.
#if defined(_WIN32) && (defined(_M_X64) || defined(_M_AMD64))

#  include <cstdio>
#  include <cstring>

#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
// dbghelp.h must follow windows.h.
#  include <dbghelp.h>

namespace
{
// dbghelp is single-threaded; serialize and only let the first faulting thread
// report, so a secondary fault while unwinding cannot interleave or recurse.
CRITICAL_SECTION g_report_lock;
LONG g_reported = 0;

// These two have no EXCEPTION_* spelling in winnt.h. ntstatus.h would name them,
// but including it alongside windows.h requires defining WIN32_NO_STATUS, which
// suppresses the STATUS_* definitions that the EXCEPTION_* macros above are
// themselves defined in terms of, and leaves NTSTATUS undeclared. Naming them
// here is cheaper than that trade.
constexpr DWORD kStatusStackBufferOverrun = 0xC0000409; // STATUS_STACK_BUFFER_OVERRUN, also __fastfail
constexpr DWORD kStatusHeapCorruption     = 0xC0000374; // STATUS_HEAP_CORRUPTION

[[nodiscard]] bool is_fatal(DWORD code) noexcept
{
  switch (code)
  {
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_ILLEGAL_INSTRUCTION:
    case EXCEPTION_PRIV_INSTRUCTION:
    case EXCEPTION_IN_PAGE_ERROR:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_DATATYPE_MISALIGNMENT:
    case kStatusStackBufferOverrun:
    case kStatusHeapCorruption:
      return true;
    default:
      return false;
  }
}

//! Module name plus offset, resolved without dbghelp so it still works when no
//! symbols are available.
void print_module_offset(void* addr) noexcept
{
  HMODULE module = nullptr;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         static_cast<LPCSTR>(addr),
                         &module)
      && module != nullptr)
  {
    char path[MAX_PATH] = {};
    if (GetModuleFileNameA(module, path, MAX_PATH) != 0)
    {
      const char* const base = std::strrchr(path, '\\');
      std::fprintf(
        stderr,
        "%s+0x%llx",
        base ? base + 1 : path,
        static_cast<unsigned long long>(static_cast<unsigned char*>(addr) - reinterpret_cast<unsigned char*>(module)));
      return;
    }
  }
  std::fprintf(stderr, "<unknown module>");
}

void print_stack(CONTEXT* context) noexcept
{
  const HANDLE process = GetCurrentProcess();
  const HANDLE thread  = GetCurrentThread();

  // SYMOPT_DEFERRED_LOADS is deliberately not set: it makes SymFromAddr fail to
  // resolve anything here, which defeats the point of this handler.
  SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_FAIL_CRITICAL_ERRORS | SYMOPT_NO_PROMPTS);
  if (!SymInitialize(process, nullptr, TRUE))
  {
    // Another component in this process -- LLVM inside libnvcc also uses
    // DbgHelp -- may already own the session. Reuse it, but pick up anything
    // loaded since it was created.
    if (!SymRefreshModuleList(process))
    {
      std::fprintf(stderr, "[cccl-crash]   (no DbgHelp session: %lu; frames show module+offset only)\n", GetLastError());
    }
  }

  // StackWalk64 mutates the context it is given.
  CONTEXT walk_context = *context;
  STACKFRAME64 frame{};
  frame.AddrPC.Offset    = walk_context.Rip;
  frame.AddrPC.Mode      = AddrModeFlat;
  frame.AddrFrame.Offset = walk_context.Rbp;
  frame.AddrFrame.Mode   = AddrModeFlat;
  frame.AddrStack.Offset = walk_context.Rsp;
  frame.AddrStack.Mode   = AddrModeFlat;

  alignas(SYMBOL_INFO) unsigned char symbol_storage[sizeof(SYMBOL_INFO) + 1024] = {};
  auto* const symbol   = reinterpret_cast<SYMBOL_INFO*>(symbol_storage);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
  symbol->MaxNameLen   = 1024;

  for (int i = 0; i < 64; ++i)
  {
    if (!StackWalk64(
          IMAGE_FILE_MACHINE_AMD64,
          process,
          thread,
          &frame,
          &walk_context,
          nullptr,
          SymFunctionTableAccess64,
          SymGetModuleBase64,
          nullptr)
        || frame.AddrPC.Offset == 0)
    {
      break;
    }

    auto* const pc = reinterpret_cast<void*>(frame.AddrPC.Offset);
    std::fprintf(stderr, "[cccl-crash]   %2d  ", i);
    print_module_offset(pc);

    DWORD64 displacement = 0;
    if (SymFromAddr(process, frame.AddrPC.Offset, &displacement, symbol))
    {
      std::fprintf(stderr, "  %s+0x%llx", symbol->Name, static_cast<unsigned long long>(displacement));
    }

    IMAGEHLP_LINE64 line{};
    line.SizeOfStruct       = sizeof(line);
    DWORD line_displacement = 0;
    if (SymGetLineFromAddr64(process, frame.AddrPC.Offset, &line_displacement, &line))
    {
      std::fprintf(stderr, "  [%s:%lu]", line.FileName, line.LineNumber);
    }
    std::fprintf(stderr, "\n");
  }
  std::fflush(stderr);
}

LONG CALLBACK crash_handler(EXCEPTION_POINTERS* info) noexcept
{
  const EXCEPTION_RECORD* const record = info ? info->ExceptionRecord : nullptr;
  if (record == nullptr || !is_fatal(record->ExceptionCode))
  {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  if (InterlockedCompareExchange(&g_reported, 1, 0) != 0)
  {
    return EXCEPTION_CONTINUE_SEARCH;
  }

  if (record->ExceptionCode == EXCEPTION_STACK_OVERFLOW)
  {
    // Almost no stack remains here, and the faulting thread may be an LLVM
    // worker with no stack guarantee reserved. Write a static message straight
    // through WriteFile: no formatting, no stack buffer, no DbgHelp, no CRT
    // buffering and no lock, so nothing here can fault again. The faulting
    // address is deliberately omitted; formatting it would need the stack we
    // have just run out of, and for a stack overflow it says little anyway.
    static const char message[] = "\n[cccl-crash] stack overflow (stack walk skipped)\n";
    DWORD written               = 0;
    WriteFile(GetStdHandle(STD_ERROR_HANDLE), message, static_cast<DWORD>(sizeof(message) - 1), &written, nullptr);
    return EXCEPTION_CONTINUE_SEARCH;
  }

  EnterCriticalSection(&g_report_lock);

  std::fprintf(stderr, "\n[cccl-crash] ===== native fault, see NVIDIA/cccl#10802 =====\n");
  std::fprintf(stderr,
               "[cccl-crash] code=0x%08lX  address=0x%p  thread=%lu\n",
               record->ExceptionCode,
               record->ExceptionAddress,
               GetCurrentThreadId());

  std::fprintf(stderr, "[cccl-crash] faulting site: ");
  print_module_offset(record->ExceptionAddress);
  std::fprintf(stderr, "\n");

  if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION && record->NumberParameters >= 2)
  {
    const ULONG_PTR operation = record->ExceptionInformation[0];
    std::fprintf(
      stderr,
      "[cccl-crash] %s address 0x%llx\n",
      operation == 0   ? "read from"
      : operation == 1 ? "write to"
                       : "execute at",
      static_cast<unsigned long long>(record->ExceptionInformation[1]));
  }

  if (info->ContextRecord != nullptr)
  {
    std::fprintf(stderr, "[cccl-crash] stack:\n");
    print_stack(info->ContextRecord);
  }

  std::fprintf(stderr, "[cccl-crash] ===== end =====\n");
  std::fflush(stderr);

  LeaveCriticalSection(&g_report_lock);

  // Let Catch2 and ctest report exactly as before; this output is additive.
  return EXCEPTION_CONTINUE_SEARCH;
}

struct crash_handler_installer
{
  crash_handler_installer()
  {
    InitializeCriticalSection(&g_report_lock);
    // First in the vectored chain, so the report is produced before Catch2's
    // fatal-condition handler unwinds the state we want to see.
    AddVectoredExceptionHandler(1, crash_handler);
  }
};

const crash_handler_installer g_installer{};
} // namespace

#endif // defined(_WIN32) && (defined(_M_X64) || defined(_M_AMD64))
