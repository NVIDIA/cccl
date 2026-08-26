Param(
    # PowerShell matches parameter names case-insensitively, so -jobs / -iterations
    # / -seconds bind here without aliases (an alias equal to the parameter name is
    # rejected outright).
    [int]$Jobs = 1,
    [int]$Iterations = 100000,
    [int]$Seconds = 900
)

$ErrorActionPreference = "Stop"

Import-Module "$PSScriptRoot/build_common.psm1"

# Resolved directly rather than via Get-RepoRoot, which lives in the Python
# helper module this job has no other reason to import.
$repoRoot = (Resolve-Path "$PSScriptRoot/../..").Path
$source = Join-Path $repoRoot "ci/nvjitlink_repro/njl_repro.cpp"
$workDir = Join-Path $repoRoot "njl-repro"
New-Item -ItemType Directory -Force -Path $workDir | Out-Null

$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) { throw "CUDA_PATH is not set" }
Write-Host "CUDA_PATH=$cudaPath"
& "$cudaPath\bin\nvcc.exe" --version | Select-String "release"

Push-Location $workDir
try {
    Invoke-Checked {
        & cl /nologo /std:c++17 /EHsc /O2 "/I$cudaPath\include" $source `
            /link "$cudaPath\lib\x64\nvrtc.lib" "$cudaPath\lib\x64\nvJitLink.lib" "$cudaPath\lib\x64\cuda.lib"
    } "failed to build njl_repro.cpp"

    $exe = Join-Path $workDir "njl_repro.exe"

    # Capture a minidump if the driver fail-fasts: it unwinds no handlers, so the
    # exit code and the Windows crash record are the only evidence available.
    $crashDumpDir = Join-Path $repoRoot "crash-dumps"
    Enable-CrashDumps $crashDumpDir @("njl_repro.exe")
    $runStart = Get-Date

    Write-Host "=== running $Jobs process(es), up to $Iterations cycles / ${Seconds}s each ==="
    $procs = @()
    for ($i = 0; $i -lt $Jobs; $i++) {
        $out = Join-Path $workDir "repro_$i.log"
        $procs += Start-Process -FilePath $exe -ArgumentList $Iterations, $Seconds `
            -PassThru -NoNewWindow -RedirectStandardOutput $out -RedirectStandardError "$out.err"
    }
    $procs | Wait-Process

    $failed = 0
    for ($i = 0; $i -lt $Jobs; $i++) {
        $code = $procs[$i].ExitCode
        Write-Host "--- process $i exit: $(Format-ExitCode $code) ---"
        Get-Content (Join-Path $workDir "repro_$i.log") -Tail 3 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "    $_" }
        Get-Content (Join-Path $workDir "repro_$i.log.err") -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "    stderr: $_" }
        if ($code -ne 0) { $failed++ }
    }

    Show-CrashDiagnostics $runStart $crashDumpDir

    if ($failed -gt 0) {
        throw "$failed of $Jobs reproducer process(es) terminated abnormally"
    }
    Write-Host "all $Jobs process(es) completed cleanly"
}
finally { Pop-Location }
