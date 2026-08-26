<#
.SYNOPSIS
    Entry point for the Windows cuda.compute test lanes.
.DESCRIPTION
    Provisions the cuda_cccl wheel and then runs the test payload, by default in
    a minimal sibling container (see run_in_minimal_container.ps1 for what that
    buys). `sysctk` is the exception: that mode exists to test against a
    system-provided CUDA toolkit, which only the devcontainer has. Set
    CCCL_MINIMAL_CONTAINER=0 to stay in the devcontainer as well -- useful
    locally, or to compare against the devcontainer environment.
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

# Fetching the wheel needs gh and the workflow helpers, which the minimal
# container deliberately does not have -- so it happens out here.
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
