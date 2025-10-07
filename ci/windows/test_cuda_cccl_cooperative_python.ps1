Param(
    [Parameter(Mandatory = $true)]
    [Alias("py-version")]
    [ValidatePattern("^\d+\.\d+$")]
    [string]$PyVersion
)

$ErrorActionPreference = "Stop"

# Import shared helpers
Import-Module "$PSScriptRoot/build_common.psm1"

$python = Get-Python -Version $PyVersion
$cudaMajor = Get-CudaMajor
