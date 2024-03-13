# This effectively does a `find_package` actually going through the find_package
# machinery. Using `find_package` works for the first configure, but creates
# inconsistencies during subsequent configurations when using CPM..
#
# More details are in the discussion at
# https://github.com/NVIDIA/libcudacxx/pull/242#discussion_r794003857
include(${CudaNext_SOURCE_DIR}/lib/cmake/CudaNext/cudanext-config-version.cmake)
include(${CudaNext_SOURCE_DIR}/lib/cmake/CudaNext/cudanext-config.cmake)
