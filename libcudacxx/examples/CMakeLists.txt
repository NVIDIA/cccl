cmake_minimum_required(VERSION 3.18)

project(libcudacxx-examples LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# Download input files for the trie examples.
if(NOT (EXISTS books))
    execute_process(COMMAND mkdir books)
    file(DOWNLOAD https://www.gutenberg.org/files/2600/2600-0.txt books/2600-0.txt SHOW_PROGRESS)
    file(DOWNLOAD http://www.gutenberg.org/cache/epub/996/pg996.txt books/pg996.txt SHOW_PROGRESS)
    file(DOWNLOAD http://www.gutenberg.org/cache/epub/55/pg55.txt books/pg55.txt SHOW_PROGRESS)
    file(DOWNLOAD https://www.gutenberg.org/files/8800/8800.txt books/8800.txt SHOW_PROGRESS)
    file(DOWNLOAD https://www.gutenberg.org/files/84/84-0.txt books/84-0.txt SHOW_PROGRESS)
    file(DOWNLOAD http://www.gutenberg.org/cache/epub/6130/pg6130.txt books/pg6130.txt SHOW_PROGRESS)
    file(DOWNLOAD http://www.gutenberg.org/cache/epub/1727/pg1727.txt books/pg1727.txt SHOW_PROGRESS)
    file(DOWNLOAD https://www.gutenberg.org/files/2701/2701-0.txt books/2701-0.txt SHOW_PROGRESS)
    file(DOWNLOAD https://www.gutenberg.org/files/35/35-0.txt books/35-0.txt SHOW_PROGRESS)
    file(DOWNLOAD https://www.gutenberg.org/files/1342/1342-0.txt books/1342-0.txt SHOW_PROGRESS)
endif()

add_executable(trie_st trie_st.cpp)
target_compile_features(trie_st PRIVATE cxx_std_11)

add_executable(trie_mt trie_mt.cpp)
target_compile_features(trie_mt PRIVATE cxx_std_11)
target_link_libraries(trie_mt Threads::Threads)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 11.1)
    add_executable(trie_cuda trie.cu)
    target_compile_features(trie_cuda PRIVATE cxx_std_11 cuda_std_11)
    target_compile_options(trie_cuda PRIVATE --expt-relaxed-constexpr)
    set_property(TARGET trie_cuda PROPERTY CUDA_ARCHITECTURES 70)
else()
    message(STATUS "Insufficient CUDA version. Skipping trie.cu example.")
endif()

if(CUDAToolkit_VERSION VERSION_GREATER 10.2)
    add_executable(rtc rtc_example.cpp)
    target_link_libraries(rtc CUDA::nvrtc)
    target_compile_features(rtc PRIVATE cxx_std_11)
else()
    message(STATUS "Insufficient CUDA version. Skipping rtc_example.cpp example.")
endif()

add_executable(hash_map concurrent_hash_table.cu)
target_compile_features(hash_map PRIVATE cxx_std_14 cuda_std_14)
set_property(TARGET hash_map PROPERTY CUDA_ARCHITECTURES 70)
target_compile_options(hash_map PRIVATE --expt-extended-lambda)
