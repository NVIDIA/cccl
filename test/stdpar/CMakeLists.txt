cmake_minimum_required(VERSION 3.21)

# NOTE: this is build outside of the libcu++ test harness
project(CCCL_STDPAR_TESTS LANGUAGES CXX)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
  message(FATAL_ERROR "The stdpar tests require nvc++ for CMAKE_CXX_COMPILER.")
endif()

# Enable testing for the project
enable_testing()

find_package(
  CCCL
  CONFIG
  REQUIRED
  NO_DEFAULT_PATH # Only check the explicit HINTS below:
  HINTS "${CMAKE_CURRENT_LIST_DIR}/../../lib/cmake/cccl/"
)

file(
  GLOB test_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  "tests/*.cpp"
)

function(cccl_add_stdpar_test test_file)
  get_filename_component(test_name ${test_file} NAME_WE)

  add_executable(stdpar_test_${test_name} ${test_file})
  target_link_libraries(stdpar_test_${test_name} PUBLIC CCCL::CCCL)

  # Ensure that we are testing with GPU support
  target_compile_options(stdpar_test_${test_name} PUBLIC -stdpar=gpu)
  target_link_options(stdpar_test_${test_name} PUBLIC -stdpar=gpu)

  # Ensure that we are indeed testing the same CCCL version
  target_compile_definitions(
    stdpar_test_${test_name}
    PUBLIC
      CMAKE_CCCL_VERSION_MAJOR=${CCCL_VERSION_MAJOR}
      CMAKE_CCCL_VERSION_MINOR=${CCCL_VERSION_MINOR}
      CMAKE_CCCL_VERSION_PATCH=${CCCL_VERSION_PATCH}
  )

  # Register with ctest
  add_test(NAME stdpar_test_${test_name} COMMAND stdpar_test_${test_name})
endfunction()

foreach (test IN LISTS test_files)
  cccl_add_stdpar_test(${test})
endforeach()
