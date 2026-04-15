# For every public header, build a translation unit containing `#include <header>`
# to let the compiler try to figure out warnings in that header if it is not otherwise
# included in tests, and also to verify if the headers are modular enough.
# .inl files are not globbed for, because they are not supposed to be used as public
# entrypoints.

cccl_get_cudatoolkit()

# Meta target for all configs' header builds:
add_custom_target(libcudacxx.test.internal_headers)

# Grep all internal headers
file(
  GLOB_RECURSE internal_headers
  RELATIVE "${libcudacxx_SOURCE_DIR}/include/"
  CONFIGURE_DEPENDS
  ${libcudacxx_SOURCE_DIR}/include/cuda/__*/*.h
  ${libcudacxx_SOURCE_DIR}/include/cuda/std/__*/*.h
)

# Exclude <cuda/std/__cccl/(prologue|epilogue|visibility).h> from the test
list(
  FILTER internal_headers
  EXCLUDE
  REGEX "__cccl/(prologue|epilogue|visibility)\.h"
)

# headers in `__cuda` are meant to come after the related "cuda" headers so they do not compile on their own
list(FILTER internal_headers EXCLUDE REGEX "__cuda/*")

# generated cuda::ptx headers are not standalone
list(FILTER internal_headers EXCLUDE REGEX "__ptx/instructions/generated")

if (CCCL_ENABLE_TILE)
  list(
    # error: asm statement is unsupported in tile code
    REMOVE_ITEM internal_headers
    "cuda/__annotated_ptr/access_property.h"
    "cuda/__annotated_ptr/access_property_encoding.h"
    "cuda/__annotated_ptr/apply_access_property.h"
    "cuda/__annotated_ptr/annotated_ptr.h"
    "cuda/__annotated_ptr/annotated_ptr_base.h"
    "cuda/__annotated_ptr/associate_access_property.h"
    "cuda/__atomic/atomic.h"
    "cuda/__barrier/barrier.h"
    "cuda/__barrier/barrier_arrive_tx.h"
    "cuda/__barrier/barrier_block_scope.h"
    "cuda/__barrier/barrier_expect_tx.h"
    "cuda/__barrier/barrier_thread_scope.h"
    "cuda/__container/buffer.h"
    "cuda/__container/make_buffer_with_pool.h"
    "cuda/__latch/latch.h"
    "cuda/__memcpy_async/cp_async_bulk_shared_global.h"
    "cuda/__memcpy_async/cp_async_shared_global.h"
    "cuda/__memcpy_async/dispatch_memcpy_async.h"
    "cuda/__memcpy_async/elect_one.h"
    "cuda/__memcpy_async/is_local_smem_barrier.h"
    "cuda/__memcpy_async/memcpy_async.h"
    "cuda/__memcpy_async/memcpy_async_barrier.h"
    "cuda/__memcpy_async/memcpy_async_tx.h"
    "cuda/__memcpy_async/memcpy_completion.h"
    "cuda/__memcpy_async/try_get_barrier_handle.h"
    "cuda/__memory/discard_memory.h"
    "cuda/__memory_resource/shared_resource.h"
    "cuda/__semaphore/counting_semaphore.h"
    "cuda/std/__atomic/api/common.h"
    "cuda/std/__atomic/api/owned.h"
    "cuda/std/__atomic/api/reference.h"
    "cuda/std/__atomic/functions.h"
    "cuda/std/__atomic/functions/cuda_local.h"
    "cuda/std/__atomic/functions/cuda_ptx_derived.h"
    "cuda/std/__atomic/functions/cuda_ptx_generated.h"
    "cuda/std/__atomic/types.h"
    "cuda/std/__atomic/types/base.h"
    "cuda/std/__atomic/types/common.h"
    "cuda/std/__atomic/types/locked.h"
    "cuda/std/__atomic/types/reference.h"
    "cuda/std/__atomic/types/small.h"
    "cuda/std/__atomic/wait/notify_wait.h"
    "cuda/std/__atomic/wait/polling.h"
    "cuda/std/__barrier/barrier.h"
    "cuda/std/__latch/latch.h"
    "cuda/std/__pstl/copy.h"
    "cuda/std/__pstl/copy_if.h"
    "cuda/std/__pstl/copy_n.h"
    "cuda/std/__pstl/count.h"
    "cuda/std/__pstl/count_if.h"
    "cuda/std/__pstl/cuda/copy_if.h"
    "cuda/std/__pstl/cuda/copy_n.h"
    "cuda/std/__pstl/cuda/exclusive_scan.h"
    "cuda/std/__pstl/cuda/generate_n.h"
    "cuda/std/__pstl/cuda/inclusive_scan.h"
    "cuda/std/__pstl/cuda/merge.h"
    "cuda/std/__pstl/cuda/partition.h"
    "cuda/std/__pstl/cuda/partition_copy.h"
    "cuda/std/__pstl/cuda/reduce.h"
    "cuda/std/__pstl/cuda/remove_if.h"
    "cuda/std/__pstl/cuda/rotate.h"
    "cuda/std/__pstl/cuda/rotate_copy.h"
    "cuda/std/__pstl/cuda/transform.h"
    "cuda/std/__pstl/cuda/transform_reduce.h"
    "cuda/std/__pstl/cuda/unique.h"
    "cuda/std/__pstl/exclusive_scan.h"
    "cuda/std/__pstl/fill.h"
    "cuda/std/__pstl/fill_n.h"
    "cuda/std/__pstl/generate.h"
    "cuda/std/__pstl/generate_n.h"
    "cuda/std/__pstl/inclusive_scan.h"
    "cuda/std/__pstl/merge.h"
    "cuda/std/__pstl/partition.h"
    "cuda/std/__pstl/partition_copy.h"
    "cuda/std/__pstl/reduce.h"
    "cuda/std/__pstl/remove.h"
    "cuda/std/__pstl/remove_copy.h"
    "cuda/std/__pstl/remove_copy_if.h"
    "cuda/std/__pstl/remove_if.h"
    "cuda/std/__pstl/replace.h"
    "cuda/std/__pstl/replace_copy.h"
    "cuda/std/__pstl/replace_copy_if.h"
    "cuda/std/__pstl/replace_if.h"
    "cuda/std/__pstl/reverse.h"
    "cuda/std/__pstl/reverse_copy.h"
    "cuda/std/__pstl/rotate.h"
    "cuda/std/__pstl/rotate_copy.h"
    "cuda/std/__pstl/swap_ranges.h"
    "cuda/std/__pstl/transform.h"
    "cuda/std/__pstl/transform_exclusive_scan.h"
    "cuda/std/__pstl/transform_inclusive_scan.h"
    "cuda/std/__pstl/transform_reduce.h"
    "cuda/std/__pstl/unique.h"
    "cuda/std/__pstl/unique_copy.h"
    "cuda/std/__semaphore/atomic_semaphore.h"
    "cuda/std/__semaphore/counting_semaphore.h"
  )

  list(
    # error: global scope non-placement dynamic deallocation with operator delete is unsupported in tile code
    REMOVE_ITEM internal_headers
    "cuda/std/__random/seed_seq.h"
  )

  list(
    # error: bit field read/write is unsupported in tile code
    REMOVE_ITEM internal_headers
    "cuda/std/__format/format_integral.h"
    "cuda/std/__format/format_spec_parser.h"
    "cuda/std/__format/output_utils.h"
    "cuda/std/__format/formatters/bool.h"
    "cuda/std/__format/formatters/char.h"
    "cuda/std/__format/formatters/int.h"
    "cuda/std/__format/formatters/fp.h"
    "cuda/std/__format/formatters/ptr.h"
    "cuda/std/__format/formatters/str.h"
  )

  list(
    # error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code
    REMOVE_ITEM internal_headers
    "cuda/__annotated_ptr/annotated_ptr.h"
    "cuda/__container/buffer.h"
    "cuda/__memcpy_async/cp_async_bulk_shared_global.h"
    "cuda/__memcpy_async/dispatch_memcpy_async.h"
    "cuda/__memcpy_async/elect_one.h"
    "cuda/__memcpy_async/memcpy_async.h"
    "cuda/__memcpy_async/memcpy_async_barrier.h"
    "cuda/std/__pstl/copy.h"
    "cuda/std/__pstl/copy_if.h"
    "cuda/std/__pstl/copy_n.h"
    "cuda/std/__pstl/count.h"
    "cuda/std/__pstl/count_if.h"
    "cuda/std/__pstl/cuda/copy_if.h"
    "cuda/std/__pstl/cuda/copy_n.h"
    "cuda/std/__pstl/cuda/exclusive_scan.h"
    "cuda/std/__pstl/cuda/generate_n.h"
    "cuda/std/__pstl/cuda/inclusive_scan.h"
    "cuda/std/__pstl/cuda/partition.h"
    "cuda/std/__pstl/cuda/partition_copy.h"
    "cuda/std/__pstl/cuda/reduce.h"
    "cuda/std/__pstl/cuda/remove_if.h"
    "cuda/std/__pstl/cuda/transform.h"
    "cuda/std/__pstl/cuda/transform_reduce.h"
    "cuda/std/__pstl/cuda/unique.h"
    "cuda/std/__pstl/exclusive_scan.h"
    "cuda/std/__pstl/fill.h"
    "cuda/std/__pstl/fill_n.h"
    "cuda/std/__pstl/generate.h"
    "cuda/std/__pstl/generate_n.h"
    "cuda/std/__pstl/inclusive_scan.h"
    "cuda/std/__pstl/partition.h"
    "cuda/std/__pstl/partition_copy.h"
    "cuda/std/__pstl/reduce.h"
    "cuda/std/__pstl/remove.h"
    "cuda/std/__pstl/remove_copy.h"
    "cuda/std/__pstl/remove_copy_if.h"
    "cuda/std/__pstl/remove_if.h"
    "cuda/std/__pstl/replace.h"
    "cuda/std/__pstl/replace_copy.h"
    "cuda/std/__pstl/replace_copy_if.h"
    "cuda/std/__pstl/replace_if.h"
    "cuda/std/__pstl/reverse.h"
    "cuda/std/__pstl/reverse_copy.h"
    "cuda/std/__pstl/swap_ranges.h"
    "cuda/std/__pstl/transform.h"
    "cuda/std/__pstl/transform_exclusive_scan.h"
    "cuda/std/__pstl/transform_inclusive_scan.h"
    "cuda/std/__pstl/transform_reduce.h"
    "cuda/std/__pstl/unique.h"
    "cuda/std/__pstl/unique_copy.h"
  )

  list(
    # error: indirect call is unsupported in tile code
    REMOVE_ITEM internal_headers
    "cuda/__annotated_ptr/annotated_ptr.h"
    "cuda/__barrier/barrier_arrive_tx.h"
    "cuda/__barrier/barrier_block_scope.h"
    "cuda/__barrier/barrier_expect_tx.h"
    "cuda/__barrier/barrier_thread_scope.h"
    "cuda/__memcpy_async/try_get_barrier_handle.h"
    "cuda/__memcpy_async/memcpy_async.h"
    "cuda/__memcpy_async/memcpy_async_barrier.h"
    "cuda/__memcpy_async/memcpy_async_tx.h"
    "cuda/__memcpy_async/memcpy_completion.h"
  )
endif()

function(libcudacxx_add_internal_header_test_target target_name)
  if (NOT ARGN)
    return()
  endif()

  cccl_generate_header_tests(
    ${target_name}
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${ARGN}
  )

  target_compile_definitions(${target_name} PRIVATE _CCCL_HEADER_TEST)
  target_link_libraries(
    ${target_name}
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
  add_dependencies(libcudacxx.test.internal_headers ${target_name})
endfunction()

libcudacxx_add_internal_header_test_target(
  libcudacxx.test.internal_headers.base
  ${internal_headers}
)

# We have fallbacks for some type traits that we want to explicitly test so that they do not bitrot.
set(internal_headers_fallback)
set(internal_headers_fallback_per_header_defines)
foreach (header IN LISTS internal_headers)
  # MSVC cannot handle some of the fallbacks.
  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (
      "${header}" MATCHES "is_base_of"
      OR "${header}" MATCHES "is_nothrow_destructible"
      OR "${header}" MATCHES "is_polymorphic"
    )
      continue()
    endif()
  endif()

  file(READ "${libcudacxx_SOURCE_DIR}/include/${header}" header_file)
  string(REGEX MATCH "_LIBCUDACXX_[A-Z_]*_FALLBACK" fallback "${header_file}")
  if (fallback)
    list(APPEND internal_headers_fallback "${header}")
    string(
      REGEX REPLACE
      "([][+.*^$()|?\\\\])"
      "\\\\\\1"
      header_regex
      "${header}"
    )
    list(
      APPEND internal_headers_fallback_per_header_defines
      DEFINE
      "${fallback}"
      "^${header_regex}$"
    )
  endif()
endforeach()

if (internal_headers_fallback)
  cccl_generate_header_tests(
    libcudacxx.test.internal_headers.fallback
    libcudacxx/include
    NO_METATARGETS
    LANGUAGE CUDA
    HEADER_TEMPLATE "${libcudacxx_SOURCE_DIR}/cmake/header_test.cpp.in"
    HEADERS ${internal_headers_fallback}
    PER_HEADER_DEFINES ${internal_headers_fallback_per_header_defines}
  )
  target_compile_definitions(
    libcudacxx.test.internal_headers.fallback
    PRIVATE _CCCL_HEADER_TEST
  )
  target_link_libraries(
    libcudacxx.test.internal_headers.fallback
    PUBLIC #
      libcudacxx.compiler_interface
      CUDA::cudart
  )
  add_dependencies(
    libcudacxx.test.internal_headers
    libcudacxx.test.internal_headers.fallback
  )
endif()
