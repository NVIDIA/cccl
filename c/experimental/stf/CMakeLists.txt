cmake_minimum_required(VERSION 3.21)

project(CCCL_C_EXPERIMENTAL_STF LANGUAGES CUDA CXX C)

option(CCCL_C_EXPERIMENTAL_STF_ENABLE_TESTING "Build cccl.experimental.c.stf tests." OFF)

# FIXME Ideally this would be handled by presets and install rules, but for now
# consumers may override this to control the target location of cccl.c.experimental.stf.
set(CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY "" CACHE PATH "Override output directory for the cccl.c.experimental.stf library")
mark_as_advanced(CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY)

file(GLOB_RECURSE srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  "src/*.cu" "src/*.cuh"
)

add_library(cccl.c.experimental.stf SHARED ${srcs})
set_property(TARGET cccl.c.experimental.stf PROPERTY POSITION_INDEPENDENT_CODE ON)
cccl_configure_target(cccl.c.experimental.stf DIALECT 20)

# Override the properties set by cccl_configure_target:
if (CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY)
  set_target_properties(cccl.c.experimental.stf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY "${CCCL_C_EXPERIMENTAL_STF_LIBRARY_OUTPUT_DIRECTORY}"
  )
endif()

find_package(cudax REQUIRED CONFIG
  NO_DEFAULT_PATH # Only check the explicit path in HINTS:
  HINTS "${CCCL_SOURCE_DIR}/lib/cmake/cudax/"
)

find_package(CUDAToolkit REQUIRED)
set_target_properties(cccl.c.experimental.stf PROPERTIES CUDA_RUNTIME_LIBRARY STATIC)
target_compile_definitions(cccl.c.experimental.stf PUBLIC CCCL_C_EXPERIMENTAL=1)
target_link_libraries(cccl.c.experimental.stf PRIVATE
  CUDA::cudart_static
  CUDA::cuda_driver
  CCCL::cudax
)

target_compile_options(cccl.c.experimental.stf PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
target_compile_options(cccl.c.experimental.stf PRIVATE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--extended-lambda>)

target_include_directories(cccl.c.experimental.stf PUBLIC "include")
target_include_directories(cccl.c.experimental.stf PRIVATE "src")

if (CCCL_C_EXPERIMENTAL_STF_ENABLE_TESTING)
  add_subdirectory(test)
endif()
