# CUDA runtime smoke test — registered as CTest fixture cccl.cuda_runtime_ok
# add_test() override gates all other tests on this fixture

enable_language(CUDA)

cccl_get_c2h()
cccl_get_cudatoolkit()

set(_smoke_target cccl.test.cuda_runtime_smoke)

cccl_add_executable(${_smoke_target} SOURCES cuda_runtime_smoke.cu ADD_CTEST)

target_link_libraries(
  ${_smoke_target}
  PRIVATE cccl.c2h.main CUDA::cudart_static
)

# the top level override skips this test to avoid self dependency.
set_tests_properties(
  ${_smoke_target}
  PROPERTIES FIXTURES_SETUP cccl.cuda_runtime_ok
)
