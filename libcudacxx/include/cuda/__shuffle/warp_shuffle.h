
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
#include <cuda/__cmath/ceil_div.h>


#define _CCCL_HAS_CUDA_COMPILER 1 //fix for now -- to be deleted later

#if _CCCL_HAS_CUDA_COMPILER
_LIBCUDACXX_BEGIN_NAMESPACE_CUDA
    template <typename T>
    constexpr bool is_supported_type = cuda::std::is_trivially_copyable<T>::value;
    //Input validation for shuffle operations
    void _CCCL_DEVICE validate_width_mask(cuda::std::int32_t width, cuda::std::uint32_t mask)
    {
        _CCCL_ASSERT((width >= 0), "Width must be greater than or equal to zero"); // width must be greater than or equal to zero
        _CCCL_ASSERT((width <= warpSize), "Width must not exceed warp size"); // width must not exceed warp size
        _CCCL_ASSERT((mask & __activemask()) == mask, "Mask must be a subset of the active mask"); // mask must be a subset of __activemask()
        _CCCL_ASSERT(cuda::std::has_single_bit(width), "Width must be a power of two"); // width must be a power of two
    }

    template<typename T>
    _CCCL_DEVICE void to_32bitBuffer(T& var, cuda::std::int32_t numElements)
    {
        cuda::std::array<cuda::std::uint32_t, numElements> buffer;
        std::memcpy(buffer.data(), &var, sizeof(T));
        return buffer;
    }

    template<typename T>
    _CCCL_DEVICE T from_32bitBuffer(cuda::std::uint32_t* inArray, cuda::std::int32_t numElements)
    {
        T var;
        std::memcpy(&var, inArray, sizeof(T));
        return var;
    }

    template <typename T>
    _CCCL_DEVICE T shfl(T var, cuda::std::int32_t srcLane, cuda::std::uint32_t mask = 0xFFFFFFFF, cuda::std::int32_t width = warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_width_mask(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((srcLane >= 0 && srcLane < width), "srcLane must be in the range [0, width)"); // srcLane must be in the range [0, width)
        //scrLane must be part of mask
        _CCCL_ASSERT((mask >> srcLane) & 1, "srcLane must be part of the mask");

        cuda::std::int32_t numElements = cuda::ceil_div(sizeof(T), sizeof(cuda::std::uint32_t));
        cuda::std::array<cuda::std::uint32_t, numElements> buffer;
        buffer = to_32bitBuffer(var, numElements);
        
        #pragma unroll
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_sync(mask, buffer[i], srcLane, width);
        }    
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_up(T var, cuda::std::int32_t delta, cuda::std::uint32_t mask = 0xFFFFFFFF, cuda::std::int32_t width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_width_mask(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)

        auto laneid = cuda::ptx::get_sreg_laneid();
        auto target_lane = (laneid - delta)>0? (laneid - delta):0;
        _CCCL_ASSERT((mask >> target_lane) & 1, "TargetLane must be part of the mask");

        //implement the logic for shfl_up
        cuda::std::int32_t numElements = cuda::ceil_div(sizeof(T), sizeof(cuda::std::uint32_t));
        cuda::std::array<cuda::std::uint32_t, numElements> buffer;
        buffer = to_32bitBuffer(var, numElements);

        #pragma unroll
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_up_sync(mask, buffer[i], delta, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_down(T var, cuda::std::int32_t delta, cuda::std::uint32_t mask = 0xFFFFFFFF, cuda::std::int32_t width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_width_mask(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)
    
        auto laneid = cuda::ptx::get_sreg_laneid();
        auto target_lane = (laneid + delta)<width? (laneid + delta):width;
        _CCCL_ASSERT((mask >> target_lane) & 1, "TargetLane must be part of the mask");//Fix the error message - on demand

        //implement the logic for shfl_down
        cuda::std::int32_t numElements = cuda::ceil_div(sizeof(T), sizeof(cuda::std::uint32_t));
        cuda::std::array<cuda::std::uint32_t, numElements> buffer;
        buffer = to_32bitBuffer(var, numElements);

        #pragma unroll
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_down_sync(mask, buffer[i], delta, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

    template <typename T>
    _CCCL_DEVICE T shfl_xor(T var, cuda::std::int32_t lanemask, cuda::std::uint32_t mask = 0xFFFFFFFF, cuda::std::int32_t width=warpSize)
    {
        _CCCL_ASSERT(is_supported_type_v<T>, "T must be a supported type for warp shuffle operations"); // T must be a supported type for warp shuffle operations
        validate_width_mask(width, mask); // validate inputs (width and mask)
        _CCCL_ASSERT((delta > 0 && delta < width), "delta must be in the range (0, width)"); // delta must be in the range (0, width)
        
        auto laneid = cuda::ptx::get_sreg_laneid();
        auto clamped_val = cuda::std::clamp(laneid ^ lanemask, 0, width);
        _CCCL_ASSERT((mask >> clamped_val) & 1, "Clamped Value must be part of the mask"); //Fix: the error message- on demand

        cuda::std::int32_t numElements = cuda::ceil_div(sizeof(T), sizeof(cuda::std::uint32_t));
        cuda::std::array<cuda::std::uint32_t, numElements> buffer;
        buffer = to_32bitBuffer(var, numElements);

        #pragma unroll
        for(int i=0;i<numElements;i++)
        {
            buffer[i] = __shfl_xor_sync(mask, buffer[i], lanemask, width);
        }
        return from_32bitBuffer<T>(buffer, numElements);
    }

_LIBCUDACXX_END_NAMESPACE_CUDA
#endif // _CCCL_HAS_CUDA_COMPILER

#endif // _CUDA_FUNCTIONAL_SHUFFLE_SAFETY_H
