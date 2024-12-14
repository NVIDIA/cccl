// This file demonstrates how to use <nv/target> and how to avoid common
// pitfalls regarding compiler/dialect compatibility.

//=======================================================================================================================
#include <nv/target>

// The below are part of libcu++ and are exposed for users that would like a simpler method of targeting host/device
// code on NVCC, NVC++ or GCC/Clang/MSVC even when the NVCC compiler isn't present.

// These macros are to be used in lieu of common #if defined(__CUDA_ARCH__) statements and
// are only to be used inside of function scopes

/* Macros defined when including <nv/target> or virtually any libcu++ header
  NV_IF_TARGET(query, true, ...)        | Queries compilation mode and emits code if true                |
  NV_IF_ELSE_TARGET(query, true, false) | As above, but can also emit different code when false          |
  NV_DISPATCH_TARGET(...)               | Similar to a switch statement emitting code for multiple modes |
*/

/* Queryable properties defined by <nv/target>
Can be imagined as __CUDA_ARCH__ >= SM_XX
  NV_PROVIDES_SM_35
  NV_PROVIDES_SM_37
  NV_PROVIDES_SM_50
  NV_PROVIDES_SM_52
  NV_PROVIDES_SM_53
  NV_PROVIDES_SM_60
  NV_PROVIDES_SM_61
  NV_PROVIDES_SM_62
  NV_PROVIDES_SM_70
  NV_PROVIDES_SM_72
  NV_PROVIDES_SM_75
  NV_PROVIDES_SM_80
  NV_PROVIDES_SM_86
  NV_PROVIDES_SM_87

Similar to above, but instead __CUDA_ARCH__ == SM_XX
  NV_IS_EXACTLY_SM_35
  NV_IS_EXACTLY_SM_37
  NV_IS_EXACTLY_SM_50
  NV_IS_EXACTLY_SM_52
  NV_IS_EXACTLY_SM_53
  NV_IS_EXACTLY_SM_60
  NV_IS_EXACTLY_SM_61
  NV_IS_EXACTLY_SM_62
  NV_IS_EXACTLY_SM_70
  NV_IS_EXACTLY_SM_72
  NV_IS_EXACTLY_SM_75
  NV_IS_EXACTLY_SM_80
  NV_IS_EXACTLY_SM_86
  NV_IS_EXACTLY_SM_87

Queries whether if host or device code is being compiled
  NV_IS_HOST
  NV_IS_DEVICE

Static true/false values for fallbacks or user manipulation
  NV_ANY_TARGET
  NV_NO_TARGET
*/

//=======================================================================================================================
// NV_IF_ELSE_TARGET(query, true statement, false statement)
__host__ __device__ int my_popc(unsigned int v)
{
  // NV_IF_ELSE_TARGET accepts three arguments, a query and two statement.
  // Here we check if we're compiling for device code. This function acts as a backend for both CUDA and host CPU popc.
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    return __popc(v); // Is false, use CUDA intrinsic
    , // Notice comma signifying end of block
    return __builtin_popc(v); // Is host, use GCC builtin
  )
}
// Note the commas separating statements, if preprocessed code is written out the macro will be preprocessed into the
// below:
/*
  my_popc(unsigned int v) {
    {return __popc(v);}
  }
*/

//=======================================================================================================================
// NV_IF_TARGET(query, true statement) OR NV_IF_TARGET(q, t, ...)
__host__ __device__ void some_algorithm()
{
  // NV_IF_TARGET accepts two arguments, a query and a statement. (and an optional false statement in >=C++11)
  NV_IF_TARGET(NV_IS_DEVICE,
               do_device_specific_work(); // Code only emitted if compiling for device
  )
}

//=======================================================================================================================
// NV_DISPATCH_TARGET(...) - Available only in C++11 and up due to variadic macros
__host__ __device__ void my_memset(void* p, uint8_t v, uint64_t c)
{
  // Target dispatch accepts pairs of queries and statements.
  // The first positive query encountered will be emitted while others are ignored.
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_80,
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async
    if (v == 0) zero_fill(p, 0, c); // zero fill using cp.async available on SM_80
    else memset(p, v, c);
    , // Notice comma signifying end of block
    NV_ANY_TARGET, // Uncoditionally use memset in other cases
    memset(p, v, c);)
}

//***********************************************************************************************************************

// # Common pitfalls:

//=======================================================================================================================
// Embedding preprocessor statements as an argument. Perform textual manipulation outside of the macro.
/*
NV_IF_TARGET(
  NV_IS_DEVICE,
// This will break immediately on most compilers
#  if defined(ENABLE_SM_80_FEATURE)
    sm80_function();
#  else
    device_function();
#  endif
)

// Instead one could write the above as:
#if defined(ENABLE_SM_80_FEATURE)
# define OPTIMAL_DEVICE_FUNCTION() sm80_function()
#else
# define OPTIMAL_DEVICE_FUNCTION() device_function()
#endif

NV_IF_TARGET(
  NV_IS_DEVICE,
  OPTIMAL_DEVICE_FUNCTION();
)
*/

//=======================================================================================================================
// Some statements may have unguarded commas, e.g. lambdas or aggregate assignment
// Supported with C++11 and up ONLY, as it requires variadic macro processing
/*
NV_IF_TARGET(
  NV_IS_DEVICE,
    ( // You may wrap a statement or series of statements with a parenthesis to guard commas from any macro machinery
      int input[] = {x, y, z...};
      my_algorithm(input);
    )
)
*/
