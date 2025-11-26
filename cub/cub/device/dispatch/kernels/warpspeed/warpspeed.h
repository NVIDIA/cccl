#pragma once

#include <cuda_runtime.h> // To hide __device__ functions from gcc, etc.

#include <warpspeed/allocators/SmemAllocator.h>
#include <warpspeed/constantAssert.h>
#include <warpspeed/makeWarpUniform.cuh>
#include <warpspeed/optimizeSmemPtr.cuh>
#include <warpspeed/SpecialRegisters.cuh>
#include <warpspeed/SyncHandler.h>
#include <warpspeed/values.h>

// TODO: rename .cuh headers to .h
#include <warpspeed/resource/SmemPhase.cuh>
#include <warpspeed/resource/SmemRef.cuh>
#include <warpspeed/resource/SmemResource.cuh>
#include <warpspeed/resource/SmemResourceRaw.cuh>
#include <warpspeed/resource/SmemStage.cuh>
#include <warpspeed/squad/Squad.h>
#include <warpspeed/squad/SquadDesc.h>
