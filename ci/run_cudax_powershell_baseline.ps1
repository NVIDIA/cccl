# Experiment-only baseline using the unmodified upstream configuration helper.
$ErrorActionPreference = 'Stop'
$env:CCCL_BUILD_INFIX = 'powershell-baseline'
Set-Location $PSScriptRoot
Import-Module "$PSScriptRoot/windows/build_common.psm1" -ArgumentList @(20, '89', '', $false)
configure_preset 'CUDA Experimental baseline' 'cudax' '-DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20'
Set-Location "$PSScriptRoot/.."
cmake --build --preset cudax --target cudax.test.execution cudax.example.stdexec_stream -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
ctest --preset cudax --output-on-failure --output-log build/powershell-baseline/cudax/ctest.log -R '^cudax[.](test[.]execution|example[.]stdexec_stream)$'
exit $LASTEXITCODE
