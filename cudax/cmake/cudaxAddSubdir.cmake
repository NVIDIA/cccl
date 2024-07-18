# This effectively does a `find_package` actually going through the find_package
# machinery. Using `find_package` works for the first configure, but creates
# inconsistencies during subsequent configurations when using CPM..
#
# More details are in the discussion at
# https://github.com/NVIDIA/libcudacxx/pull/242#discussion_r794003857
include(${cudax_SOURCE_DIR}/lib/cmake/cudax/cudax-config-version.cmake)
include(${cudax_SOURCE_DIR}/lib/cmake/cudax/cudax-config.cmake)
