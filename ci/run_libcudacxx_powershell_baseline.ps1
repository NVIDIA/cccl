# Experiment-only baseline using the unmodified upstream configuration helper.
$ErrorActionPreference = 'Stop'
$env:CCCL_BUILD_INFIX = 'powershell-baseline'
Set-Location $PSScriptRoot
Import-Module "$PSScriptRoot/windows/build_common.psm1" -ArgumentList @(20, '89', '', $false)
configure_preset 'libcu++ baseline' 'libcudacxx' '-DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20'
Set-Location "$PSScriptRoot/.."
cmake --build --preset libcudacxx --target libcudacxx.test.cuda.memory_resource.resources.device_memory_resource libcudacxx.test.cuda.memory_resource.resources.memory_pools libcudacxx.test.cuda.memory_resource.resources.shared_memory_pools -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ctest --preset libcudacxx-ctest --output-on-failure --output-log build/powershell-baseline/libcudacxx/ctest.log -R '^libcudacxx[.]test[.]cuda[.]memory_resource[.]resources[.](device_memory_resource|memory_pools|shared_memory_pools)$'
exit $LASTEXITCODE
