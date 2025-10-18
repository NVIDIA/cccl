function Get-LatestPythonPatchVersionFromPyEnvWin {
    <#
    .SYNOPSIS
        Resolves the latest patch version for a given Python major.minor (e.g.
        '3.12') by parsing `pyenv install --list` output on Windows (pyenv-win).
    .PARAMETER Version
        A string in the form 'M.m' (e.g., '3.10', '3.11', '3.12').
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+\.\d+$')]
        [string]$Version
    )

    # Verify pyenv exists.
    if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
        throw [System.InvalidOperationException]::new(
            'pyenv-win ("pyenv") not found on PATH.'
        )
    }

    $listOutput = & pyenv install --list 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $listOutput) {
        $joined = $listOutput -join "`n"
        throw [System.InvalidOperationException]::new(
            "Failed to run 'pyenv install --list'. Output:`n$joined"
        )
    }

    # Build a list of patch numbers that match the requested minor version.
    $versionPrefix = "$Version."
    $patchNumbers = @()
    foreach ($line in $listOutput) {
        $candidate = $line.Trim()
        if (-not $candidate) { continue }

        # Accept any major version; the StartsWith check guarantees we only
        # keep the wanted minor.
        if (-not $candidate.StartsWith($versionPrefix)) { continue }
        if ($candidate -notmatch '^\d+\.\d+\.\d+$') { continue }

        $patchNumbers += [int]($candidate.Split('.')[2])
    }

    if ($patchNumbers.Count -eq 0) {
        throw [System.InvalidOperationException]::new(
            "No installable CPython versions found for prefix " +
            "'$Version' in pyenv-win list."
        )
    }

    $latestPatch = ($patchNumbers | Sort-Object -Descending)[0]
    return "$Version.$latestPatch"
}

function Install-PythonViaPyEnvWin {
    <#
    .SYNOPSIS
        Ensures a Python version for the given major.minor exists via
        pyenv-win, activates it for the current shell, and returns the
        path to python.exe.
    .PARAMETER Version
        A string in the form 'M.m' (e.g., '3.12').
    #>
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+\.\d+$')]
        [string]$Version
    )

    $fullVersion = Get-LatestPythonPatchVersionFromPyEnvWin `
        -Version $Version

    Write-Host "Installing Python $fullVersion via pyenv..."
    Write-Host "pyenv install $fullVersion"
    ($null = & pyenv install $fullVersion | Out-Host)
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new(
            "Failed to install Python $fullVersion via pyenv."
        )
    }
    Write-Host "Successfully installed Python $fullVersion via pyenv."

    ($null = & pyenv local $fullVersion | Out-Host)
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new(
            "Failed to set Python $fullVersion as local via pyenv."
        )
    }
    Write-Host "Successfully set Python $fullVersion as local via pyenv."

    # Avoid the shim (i.e. shims/python.bat) because it will attempt to set
    # a codepage via `chcp` that we probably won't have installed on our
    # Server Core-based image.
    $exe = (Resolve-Path -LiteralPath $(pyenv which python)).Path
    Write-Host "python.exe path: $exe"

    # Add the root and Scripts directory to $Env:PATH.
    $rootDir = $exe.Replace("\python.exe", "")
    $scriptsDir = $exe.Replace("python.exe", "Scripts")
    $pathPrefix = $rootDir + ";" + $scriptsDir + ";"
    $Env:PATH = $pathPrefix + $Env:PATH

    # Upgrade pip using the found exe.  This is necessary because some older
    # versions of pip (e.g. 23.10) don't support arguments like `--wheeldir`.
    ($null = & $exe -m pip install --upgrade pip --no-cache-dir | Out-Host)
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new("pip upgrade failed")
    }

    Write-Host "pip successfully upgraded, running pyenv rehash..."
    ($null = & pyenv rehash | Out-Host)
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new("pyenv rehash failed")
    }
    Write-Host "Successfully ran pyenv rehash."

    return $exe
}

function Get-Python {
    <#
    .SYNOPSIS
        Returns the path of the Python interpreter satisfying the supplied
        version, potentially installing it via pyenv-win if it's not already
        installed.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+\.\d+$')]
        [string]$Version
    )

    # Look for a plain 'python.exe' already on the path.
    try {
        $candidate = (Get-Command python -ErrorAction Stop).Source
        $foundVer = & $candidate -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($foundVer -eq $Version) {
            Write-Host "Found matching Python $foundVer at $candidate."
            return $candidate.Trim()
        }
        else {
            Write-Host "Found python.exe but version $foundVer != requested version $Version."
        }
    }
    catch {
        Write-Host "Unable to query existing 'python' on PATH: $_"
    }

    # If we reach here, we'll need to install the requested version via pyenv.
    try {
        $exe = Install-PythonViaPyEnvWin -Version $Version
        return $exe.Trim()
    }
    catch {
        throw [System.InvalidOperationException]::new(
            "Requested Python $Version not found and installation " +
            "via pyenv-win failed: $($_.Exception.Message)"
        )
    }
}

function Get-RepoRoot {
    return (Resolve-Path "$PSScriptRoot/../..")
}

function Get-CudaMajor {
    <#
    .SYNOPSIS
        Gets the CUDA major version for this container instance (e.g. '12' or
        '13').  Defaults to '13' if no match can be found.
    #>
    if ($env:CUDA_PATH) {
        $nvcc = Join-Path $env:CUDA_PATH "bin/nvcc.exe"
        if (Test-Path $nvcc) {
            $out = & $nvcc --version 2>&1
            $text = ($out -join "`n")
            if ($text -match 'release\s+(\d+)\.') { return $Matches[1] }
        }
        # Fallback: parse major from CUDA_PATH like ...\v13.0 or ...\CUDA\13
        $pathMatch = [regex]::Match($env:CUDA_PATH, 'v?(\d+)(?:\.\d+)?')
        if ($pathMatch.Success) { return $pathMatch.Groups[1].Value }
    }
    return '13'
}

function Convert-ToUnixPath {
    Param([Parameter(Mandatory = $true)][string]$p)
    return ($p -replace "\\", "/")
}

function Get-CudaCcclWheel {
    <#
    .SYNOPSIS
        Returns the path of the cuda-cccl wheel artifact to use in the context
        of a GitHub Actions CI test script.
    #>
    Param()

    $repoRoot = Get-RepoRoot
    if ($env:GITHUB_ACTIONS) {
        Push-Location $repoRoot
        try {
            $wheelArtifactName = (& bash -lc "ci/util/workflow/get_wheel_artifact_name.sh").Trim()
            if (-not $wheelArtifactName) { throw 'Failed to resolve wheel artifact name' }
            $repoRootPosix = Convert-ToUnixPath $repoRoot
            # Ensure output from downloader goes to console, not function return pipeline
            $null = (& bash -lc "ci/util/artifacts/download.sh $wheelArtifactName $repoRootPosix" 2>&1 | Out-Host)
            if ($LASTEXITCODE -ne 0) { throw "Failed to download wheel artifact '$wheelArtifactName'" }
        }
        finally { Pop-Location }
    }

    $wheelhouse = Join-Path $repoRoot 'wheelhouse'
    $wheelPath = Get-OnePathMatch -Path $wheelhouse -Pattern '^cuda_cccl-.*\.whl' -File
    return $wheelPath
}

function Get-OnePathMatch {
    <#
    .SYNOPSIS
        Returns a single path (file or directory) match for a given pattern,
        throwing an error if there were no matches or more than one match.
    #>
    [CmdletBinding(DefaultParameterSetName = 'FileSet')]
    param(
        [Parameter(Mandatory)]
        [string] $Path,

        [Parameter(Mandatory)]
        [string] $Pattern,

        [Parameter(Mandatory, ParameterSetName = 'FileSet')]
        [switch] $File,

        [Parameter(Mandatory, ParameterSetName = 'DirSet')]
        [switch] $Directory,

        [switch] $Recurse
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "Path not found or not a directory: $Path"
    }

    $gciArgs = @{
        LiteralPath = $Path
        ErrorAction = 'SilentlyContinue'
    }

    if ($Recurse) { $gciArgs['Recurse'] = $true }
    if ($PSCmdlet.ParameterSetName -eq 'FileSet') {
        $gciArgs['File'] = $true
    }
    else {
        $gciArgs['Directory'] = $true
    }

    $pathMatches = @(
        Get-ChildItem @gciArgs |
        Where-Object { $_.Name -match $Pattern } |
        Select-Object -ExpandProperty FullName
    )

    if ($pathMatches.Count -ne 1) {
        $kind = if ($PSCmdlet.ParameterSetName -eq 'FileSet') { 'file' }
        else { 'directory' }
        $indented = ($pathMatches | ForEach-Object { "    $_" }) -join "`n"

        $msg = @"
Expected exactly one $kind name matching regex:
  $Pattern
under:
  $Path
Found:
  $($pathMatches.Count)

$indented
"@
        throw $msg
    }

    return $pathMatches[0]
}

Export-ModuleMember -Function Get-Python, Get-CudaMajor, Convert-ToUnixPath, Get-RepoRoot, Get-CudaCcclWheel, Get-OnePathMatch
