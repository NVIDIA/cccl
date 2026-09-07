#pragma once

// <cuda.h> provides CUDA_VERSION, used to pick the nvJitLink symbol binding below.
#include <cuda.h>

#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#undef NVJITLINK_NO_INLINE

// nvJitLink symbol binding.
//
// With NVJITLINK_NO_INLINE the nvJitLink header leaves the plain, unversioned
// entry points (nvJitLinkCreate, ...) for us to declare. Linking those against a
// modern (>= 12.3) toolkit records *versioned* references such as
// nvJitLinkDestroy@libnvJitLink.so.12. The 12.0-12.2 libraries export only the
// versioned __nvJitLink*_12_X aliases -- no unversioned names -- so a wheel built
// with a newer 12.x toolkit fails to load on CTK 12.0-12.2 with
// "undefined symbol: nvJitLinkDestroy, version libnvJitLink.so.12".
//
// Every 12.x library retains the __nvJitLink*_12_0 aliases, so on CUDA 12 we bind
// directly to those: a wheel built with any 12.x toolkit then loads on CTK 12.0
// through the newest 12.x. CUDA 13+ exports the unversioned names, so the plain
// declarations are kept there.
#if CUDA_VERSION < 13000
#  define nvJitLinkCreate             __nvJitLinkCreate_12_0
#  define nvJitLinkDestroy            __nvJitLinkDestroy_12_0
#  define nvJitLinkAddData            __nvJitLinkAddData_12_0
#  define nvJitLinkAddFile            __nvJitLinkAddFile_12_0
#  define nvJitLinkComplete           __nvJitLinkComplete_12_0
#  define nvJitLinkGetLinkedCubinSize __nvJitLinkGetLinkedCubinSize_12_0
#  define nvJitLinkGetLinkedCubin     __nvJitLinkGetLinkedCubin_12_0
#  define nvJitLinkGetLinkedPtxSize   __nvJitLinkGetLinkedPtxSize_12_0
#  define nvJitLinkGetLinkedPtx       __nvJitLinkGetLinkedPtx_12_0
#  define nvJitLinkGetErrorLogSize    __nvJitLinkGetErrorLogSize_12_0
#  define nvJitLinkGetErrorLog        __nvJitLinkGetErrorLog_12_0
#  define nvJitLinkGetInfoLogSize     __nvJitLinkGetInfoLogSize_12_0
#  define nvJitLinkGetInfoLog         __nvJitLinkGetInfoLog_12_0
#endif

// declare unversioned functions (redirected to the _12_0 aliases on CUDA 12)

extern "C" {
nvJitLinkResult nvJitLinkCreate(nvJitLinkHandle*, uint32_t, const char**);
nvJitLinkResult nvJitLinkDestroy(nvJitLinkHandle*);
nvJitLinkResult nvJitLinkAddData(nvJitLinkHandle, nvJitLinkInputType, const void*, size_t, const char*);
nvJitLinkResult nvJitLinkAddFile(nvJitLinkHandle, nvJitLinkInputType, const char*);
nvJitLinkResult nvJitLinkComplete(nvJitLinkHandle);
nvJitLinkResult nvJitLinkGetLinkedCubinSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetLinkedCubin(nvJitLinkHandle, void*);
nvJitLinkResult nvJitLinkGetLinkedPtxSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetLinkedPtx(nvJitLinkHandle, char*);
nvJitLinkResult nvJitLinkGetErrorLogSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetErrorLog(nvJitLinkHandle, char*);
nvJitLinkResult nvJitLinkGetInfoLogSize(nvJitLinkHandle, size_t*);
nvJitLinkResult nvJitLinkGetInfoLog(nvJitLinkHandle, char*);
}
