function Get-Python {
    <#
    .SYNOPSIS
        Returns the path of the Python interpreter satisfying the supplied
        version, installing it via uv if necessary.
    .PARAMETER Version
        A string in the form 'M.m' (e.g., '3.10', '3.13').
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [ValidatePattern('^\d+\.\d+$')]
        [string]$Version
    )

    # Install uv if not present. uv downloads pre-built CPython binaries —
    # no compilation, no build dependencies, no pyenv-win required.
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv..."
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        # uv installs to $HOME\.local\bin on Windows
        $uvBin = Join-Path $HOME '.local\bin'
        $Env:PATH = $uvBin + ";" + $Env:PATH
    }

    Write-Host "Creating Python $Version venv via uv..."
    $venvDir = Join-Path $HOME '.cccl-venv'
    & uv venv --seed --python $Version $venvDir
    if ($LASTEXITCODE -ne 0) {
        throw [System.InvalidOperationException]::new(
            "Failed to create Python $Version venv via uv."
        )
    }

    $exe = Join-Path $venvDir 'Scripts\python.exe'
    if (-not (Test-Path $exe)) {
        throw [System.InvalidOperationException]::new(
            "Could not find python.exe in venv at $exe"
        )
    }

    Write-Host "Python $Version at: $exe"

    # Add venv Scripts dir to PATH so bare `python` and `pip` work.
    $scriptsDir = Join-Path $venvDir 'Scripts'
    $Env:PATH = $scriptsDir + ";" + $Env:PATH

    return $exe
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
