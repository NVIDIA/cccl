Param(
    [Parameter(Mandatory = $false)]
    [Alias("std")]
    [ValidateNotNullOrEmpty()]
    [ValidateSet(17, 20)]
    [int]$CXX_STANDARD = 17,
    [Parameter(Mandatory = $false)]
    [Alias("arch")]
    [string]$CUDA_ARCH = "",
    [Parameter(Mandatory = $false)]
    [Alias("cmake-options")]
    [string]$CMAKE_OPTIONS = "",
    [Parameter(Mandatory = $false)]
    [Alias("min-cmake")]
    [switch]$MIN_CMAKE
)

$ErrorActionPreference = "Stop"

$CURRENT_PATH = Split-Path $pwd -leaf
If($CURRENT_PATH -ne "ci") {
    Write-Host "Moving to ci folder"
    pushd "$PSScriptRoot/.."
}

if ($MIN_CMAKE) {
    $minVersion = "3.18.0"
    Write-Host "Installing minimum CMake version v$minVersion..."
    $url     = "https://github.com/Kitware/CMake/releases/download/v$minVersion/cmake-$minVersion-win64-x64.zip"
    $zipPath = Join-Path $env:TEMP "cmake-min.zip"
    $extract = Join-Path $env:TEMP "cmake-$minVersion"
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Expand-Archive -Path $zipPath -DestinationPath $extract -Force
    $ctestExe = (Get-ChildItem -Path $extract -Recurse -Filter "ctest.exe" |
                 Select-Object -First 1).FullName
    $env:MIN_CTEST_COMMAND = $ctestExe
}

Import-Module $PSScriptRoot/build_common.psm1 -ArgumentList @($CXX_STANDARD, $CUDA_ARCH, $CMAKE_OPTIONS)

$PRESET = "packaging"
$LOCAL_CMAKE_OPTIONS = ""

if ($env:GITHUB_SHA) {
    $LOCAL_CMAKE_OPTIONS = '"-DCCCL_EXAMPLE_CPM_TAG={0}"' -f $env:GITHUB_SHA
}

if ($env:MIN_CTEST_COMMAND) {
    $LOCAL_CMAKE_OPTIONS += ' "-DCCCL_EXAMPLE_CTEST_COMMAND={0}"' -f $env:MIN_CTEST_COMMAND
}

python -m pip install --quiet pyyaml

configure_preset "Packaging" $PRESET $LOCAL_CMAKE_OPTIONS
test_preset "Packaging" $PRESET

If($CURRENT_PATH -ne "ci") {
    popd
}
