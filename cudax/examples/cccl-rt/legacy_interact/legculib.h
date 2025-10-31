#ifndef LEGACY_CUDA_LIBRARY_H
#define LEGACY_CUDA_LIBRARY_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//! @brief The library error code enumeration.
typedef enum legculib_error_enum
{
  LEGCULIB_SUCCESS, //!< Operation was successful.
  LEGCULIB_ERROR_NOT_INITIALIZED, //!< Library was not initialized before calling the API.
  LEGCULIB_ERROR_INVALID_ARGUMENT, //!< An invalid argument was passed to the API call.
  LEGCULIB_ERROR_KERNEL_COMPILATION, //!< A kernel compilation failed.
  LEGCULIB_ERROR_MALLOC, //!< Failed to allocate memory from the heap.
  LEGCULIB_ERROR_CUDA_DRIVER, //!< A CUDA Driver error occurred.
} legculib_error;

//! @brief The legculib API call result.
typedef struct legculib_result
{
  legculib_error error; //!< The error status.
  cudaError_t cuda_error; //!< The CUDA Driver error code. Valid, if \c error is \c LEGCULIB_ERROR_CUDA_DRIVER.
} legculib_result;

//! @brief The enumeration of datatypes supported by the library.
typedef enum legculib_datatype
{
  LEGCULIB_I32, //!< 32-bit signed integer type.
  LEGCULIB_I64, //!< 64-bit signed integer type.
  LEGCULIB_U32, //!< 32-bit unsigned integer type.
  LEGCULIB_U64, //!< 64-bit unsigned integer type.
  LEGCULIB_F32, //!< 32-bit floating point type.
  LEGCULIB_F64, //!< 64-bit floating point type.
  LEGCULIB_DATATYPE_COUNT,
} legculib_datatype;

//! @brief Gets the error string for a given \c error.
//!
//! @param error The error to retrieve the string for.
const char* legculib_get_error_string(legculib_error error);

//! @brief Initializes the library for the given device.
//!
//! @param ordinal The ordinal device ID.
legculib_result legculib_init(int ordinal);

//! @brief Gets the device memory pool used by the legculib library.
//!
//! @param mempool_ptr Pointer to the memory pool handle to save the result in.
legculib_result legculib_get_device_mempool(cudaMemPool_t* mempool_ptr);

//! @brief Computes the sum of products.
//!
//! @param stream The stream to enqueue the computation in.
//! @param result The device pointer to store the result in.
//! @param lhs    The left hand side operands.
//! @param rhs    The right hand side operands.
//! @param n      The number of elements in \c lhs and \c rhs. If 0, then only the kernel is initialized.
//! @param type   The datatype to perform the computation in.
legculib_result legculib_sum_of_products(
  cudaStream_t stream, void* result, const void* lhs, const void* rhs, unsigned n, legculib_datatype type);

//! @brief Releases the resources owned by the library.
legculib_result legculib_finalize();

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // LEGACY_CUDA_LIBRARY_H
