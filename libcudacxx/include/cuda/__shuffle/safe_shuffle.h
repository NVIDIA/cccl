
#ifndef _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
#define _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>
#include <cuda/std/bit>
#include <cuda/std/memory>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/__ptx/instructions/generated/get_sreg.h>
#include <cuda/std/__cmath/nvfp16.h>
#include <cuda/std/__cmath/nvbf16.h>

#define _CCCL_HAS_CUDA_COMPILER 1 //fix for now -- to be deleted later

#if _CCCL_HAS_CUDA_COMPILER
_LIBCUDACXX_BEGIN_NAMESPACE_CUDA
    template <typename T>
    constexpr bool is_supported_type_v = false;
    template <> constexpr bool is_supported_type_v<int> = true;
    template <> constexpr bool is_supported_type_v<unsigned int> = true;
    template <> constexpr bool is_supported_type_v<long> = true;
    template <> constexpr bool is_supported_type_v<unsigned long> = true;
    template <> constexpr bool is_supported_type_v<long long> = true;
    template <> constexpr bool is_supported_type_v<unsigned long long> = true;
    template <> constexpr bool is_supported_type_v<float> = true;
    template <> constexpr bool is_supported_type_v<double> = true;
    #if defined(_LIBCUDACXX_HAS_NVFP16)
        template <> constexpr bool is_supported_type_v<__half> = true;
        template <> constexpr bool is_supported_type_v<__half2> = true;
    #endif
    #if defined(_LIBCUDACXX_HAS_NVBF16)
        template <> constexpr bool is_supported_type_v<__nv_bfloat16> = true;
        template <> constexpr bool is_supported_type_v<__nv_bfloat162> = true;
    #endif

    //Input validation for shuffle operations
    void _CCCL_DEVICE validate_shuffle_inputs(int width, unsigned mask)
    {
        _CCCL_ASSERT((width <= warpSize), "Width must not exceed warp size"); // width must not exceed warp size
        _CCCL_ASSERT((mask & __activemask()) == mask, "Mask must be a subset of the active mask"); // mask must be a subset of __activemask()
        _CCCL_ASSERT((width > 0 && (width & (width - 1)) == 0), "Width must be a power of two"); // width must be a power of two
    }

    template<typename T>
    _CCCL_DEVICE void to_32bitBuffer(T& var, uint32_t* outArray, int& numElements)
    {
        constexpr size_t typeSize = sizeof(T);
        constexpr int elements = (typeSize + sizeof(uint32_t) - 1) / sizeof(uint32_t);
        numElements = elements;
        std::memcpy(outArray, &var, typeSize);
    }

    template<typename T>
    _CCCL_DEVICE T from_32bitBuffer(uint32_t* inArray, int numElements)
    {
        // constexpr size_t typeSize = sizeof(T);
        T var;
        std::memcpy(&var, inArray, sizeof(T));
        return var;
    }

    template <typename T>
    _CCCL_DEVICE T shfl(T var, int srcLane, unsigned mask = 0xFFFFFFFF, int width = warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_shuffle_inputs(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((srcLane >= 0 && srcLane < width), "srcLane must be in the range [0, width)"); // srcLane must be in the range [0, width)
        //scrLane mustbe part of mask
        _CCCL_ASSERT((mask >> srcLane) & 1, "srcLane must be part of the mask");

        //implement the logic for shfl
        uint32_t buffer[sizeof(T) / sizeof(uint32_t)+1];
        int numElements;
        to_32bitBuffer(var, buffer, numElements);
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_sync(mask, buffer[i], srcLane, width);
        }    
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_up(T var, int delta, unsigned mask = 0xFFFFFFFF, int width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_shuffle_inputs(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)

        auto laneid = cuda::ptx::get_sreg_laneid();
        auto target_lane = (laneid - delta)>0? (laneid - delta):0;
        _CCCL_ASSERT((mask >> target_lane) & 1, "TargetLane must be part of the mask");

        //implement the logic for shfl_up
        uint32_t buffer[sizeof(T) / sizeof(uint32_t)+1];
        int numElements;
        to_32bitBuffer(var, buffer, numElements);
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_up_sync(mask, buffer[i], delta, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_down(T var, int delta, unsigned mask = 0xFFFFFFFF, int width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_shuffle_inputs(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)
    
        auto laneid = cuda::ptx::get_sreg_laneid();
        auto target_lane = (laneid + delta)<width? (laneid + delta):width;
        _CCCL_ASSERT((mask >> target_lane) & 1, "TargetLane must be part of the mask");//Fix the error message - on demand

        //implement the logic for shfl_down
        uint32_t buffer[sizeof(T) / sizeof(uint32_t)+1];
        int numElements;
        to_32bitBuffer(var, buffer, numElements);
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_down_sync(mask, buffer[i], delta, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_xor(T var, int lanemask, unsigned mask = 0xFFFFFFFF, int width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_shuffle_inputs(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)
        
        auto laneid = cuda::ptx::get_sreg_laneid();
        auto clamped_val = cuda::std::clamp(laneid ^ lanemask, 0, width);
        _CCCL_ASSERT((mask >> clamped_val) & 1, "Clamped Value must be part of the mask"); //Fix: the error message- on demand

        //implement the logic for shfl_xor
        uint32_t buffer[sizeof(T) / sizeof(uint32_t)+1];
        int numElements;
        to_32bitBuffer(var, buffer, numElements);
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_xor_sync(mask, buffer[i], lanemask, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

_LIBCUDACXX_END_NAMESPACE_CUDA
#endif // _CCCL_HAS_CUDA_COMPILER

#endif // _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
