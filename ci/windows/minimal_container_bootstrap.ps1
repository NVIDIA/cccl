<#
.SYNOPSIS
    Provision the minimal sibling container, then hand off to the test payload.
.DESCRIPTION
    The minimal image is deliberately bare, but "bare" is not the same as
    "unusable". `python:3.14-slim` on the Linux side still ships glibc and
    libstdc++, because every C/C++ Python extension links against them.
    `mcr.microsoft.com/windows/servercore` ships neither of the Windows
    equivalents: msvcp140.dll, vcruntime140.dll and vcruntime140_1.dll come from
    the MSVC redistributable, which is not part of Windows.

    Without them, every C++ extension in the test environment fails to load with
    "DLL load failed ... The specified module could not be found" -- numba's
    _typeconv and our own cccl.c.parallel.dll among them. That is not an
    undeclared dependency of cuda.compute that a wheel could carry: numba has the
    same requirement, python.org's own installer bundles the redistributable, and
    no amount of packaging on our side would help a machine that lacks it.

    Installing it here restores the intended comparison. The container still has
    no compiler and no CUDA toolkit, so a real undeclared dependency still fails
    loudly -- it just no longer fails for a reason that has nothing to do with
    what the lane is testing.

    The payload arrives in CCCL_MINIMAL_PAYLOAD rather than as a parameter so
    that everything on the command line is unambiguously the payload's own:
    a `Param()` block here would try to bind the payload's `-py-version` first.
#>

$ErrorActionPreference = "Stop"

$payload = $env:CCCL_MINIMAL_PAYLOAD
if (-not $payload) {
    throw "CCCL_MINIMAL_PAYLOAD is not set; nothing to run."
}
if (-not (Test-Path $payload)) {
    throw "Payload script '$payload' does not exist inside the container."
}

# Windows PowerShell 5.1 in a bare image may still default to TLS 1.0, which
# both aka.ms and astral.sh reject.
[Net.ServicePointManager]::SecurityProtocol =
    [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# msvcp140.dll is the one that is always missing; vcruntime140*.dll ship next to
# some interpreters, so they are a less reliable probe.
$msvcp = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
if (Test-Path $msvcp) {
    Write-Host "MSVC runtime already present; skipping redistributable install."
} else {
    $installer = Join-Path $env:TEMP 'vc_redist.x64.exe'
    Write-Host "Installing the MSVC runtime redistributable..."
    Invoke-WebRequest -UseBasicParsing `
        -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' `
        -OutFile $installer
    $proc = Start-Process -Wait -PassThru -FilePath $installer `
        -ArgumentList '/quiet', '/norestart'
    # 3010 is "succeeded, reboot required"; nothing here needs the reboot.
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "vc_redist.x64.exe failed with exit code $($proc.ExitCode)."
    }
    if (-not (Test-Path $msvcp)) {
        throw "vc_redist.x64.exe reported success but msvcp140.dll is still absent."
    }
}

& $payload @args
exit $LASTEXITCODE
