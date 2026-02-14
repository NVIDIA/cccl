#pragma once

#define NVJITLINK_NO_INLINE
#include <nvJitLink.h>
#undef NVJITLINK_NO_INLINE

// declare unversioned functions

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
