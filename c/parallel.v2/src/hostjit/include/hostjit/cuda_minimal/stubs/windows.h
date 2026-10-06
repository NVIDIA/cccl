// Minimal SRWLOCK stub for host JIT.
// Host JIT's include path has no Windows SDK. These are the kernel32 entry
// points recorded in the generated import library. SRWLOCK is pointer-sized.
#ifndef _HOSTJIT_WINDOWS_H
#define _HOSTJIT_WINDOWS_H

typedef struct
{
  void* Ptr;
} SRWLOCK;

typedef SRWLOCK* PSRWLOCK;

__declspec(dllimport) void __stdcall AcquireSRWLockExclusive(PSRWLOCK);
__declspec(dllimport) unsigned char __stdcall TryAcquireSRWLockExclusive(PSRWLOCK);
__declspec(dllimport) void __stdcall ReleaseSRWLockExclusive(PSRWLOCK);

#endif // _HOSTJIT_WINDOWS_H
