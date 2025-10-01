enable_language(CUDA)

#
# Architecture options:
#

option(CUB_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
option(CUB_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)
