<#
.SYNOPSIS
    Entry point for the Windows cuda.compute test lanes.
.DESCRIPTION
    Provisions the cuda_cccl wheel, then runs the test payload -- by default in a
    minimal sibling container, except in `sysctk` mode or when
    CCCL_MINIMAL_CONTAINER=0. See "Testing Python in a minimal container" in
    docs/infrastructure/ci/references/ci_overview.rst.
#>
Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+t?$")]
    [string]$PyVersion,

    [Alias("ctk-mode")]
    [string]$CtkMode = ""
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common_python.psm1"

# Needs gh and the workflow helpers, which the minimal container lacks.
$null = Get-CudaCcclWheel

$payloadArgs = @('-py-version', $PyVersion)
if ($CtkMode) { $payloadArgs += @('-ctk-mode', $CtkMode) }

if (((Get-CtkExtraFlavor $CtkMode) -ne 'sysctk') -and ($env:CCCL_MINIMAL_CONTAINER -ne '0')) {
    & "$PSScriptRoot/run_in_minimal_container.ps1" `
        -Script 'ci\windows\run_compute_tests.ps1' `
        -ScriptArgs $payloadArgs
} else {
    & "$PSScriptRoot/run_compute_tests.ps1" @payloadArgs
}
exit $LASTEXITCODE
