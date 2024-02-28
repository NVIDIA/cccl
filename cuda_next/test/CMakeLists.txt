if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90")
endif()

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.9 # or a later release
)

FetchContent_MakeAvailable(Catch2)

add_library(catch2_main STATIC catch2_helpers/catch2_main.cpp)
target_link_libraries(catch2_main Catch2::Catch2)

add_executable("hierarchy_tests" "hierarchy_smoke.cu" "hierarchy_custom_types.cu")
target_compile_options("hierarchy_tests" PRIVATE "--extended-lambda")
target_include_directories("hierarchy_tests" PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries("hierarchy_tests" Catch2::Catch2 catch2_main)
add_test(NAME "cuda_next_hierarchy_tests" COMMAND "hierarchy_tests")
