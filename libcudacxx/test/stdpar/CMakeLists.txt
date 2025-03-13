cmake_minimum_required(VERSION 3.21)

# NOTE: this is build outside of the libcu++ test harness
project(CCCL_STDPAR_TESTS LANGUAGES CXX)

# Enable testing for the project
enable_testing()

find_package(CCCL CONFIG REQUIRED
  NO_DEFAULT_PATH # Only check the explicit HINTS below:
  HINTS "${CMAKE_CURRENT_LIST_DIR}/../../../lib/cmake/cccl/"
)

file(GLOB test_files
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

  # Register with ctest
  add_test(NAME stdpar_test_${test_name} COMMAND stdpar_test_${test_name})
endfunction()

foreach(test IN LISTS test_files)
  cccl_add_stdpar_test(${test})
endforeach()
