param([ValidateSet('cpu', 'cuda')][string]$Backend = 'cuda')
$ErrorActionPreference = 'Stop'

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { throw 'Install Visual Studio Build Tools with Desktop development with C++.' }
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'MSVC x64 toolchain not found.' }
    $devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
    $toolset = ''
    if ($env:MSVC_TOOLSET_VERSION) {
        if ($env:MSVC_TOOLSET_VERSION -notmatch '^[0-9.]+$') { throw 'Invalid MSVC_TOOLSET_VERSION.' }
        $toolset = '-vcvars_ver=' + $env:MSVC_TOOLSET_VERSION
    }
    $lines = & $env:ComSpec /s /c "`"`"$devcmd`" -no_logo -arch=x64 -host_arch=x64 $toolset >nul && set`""
    if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize MSVC.' }
    foreach ($line in $lines) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
}

if ($env:VSINSTALLDIR) {
    $bundled = Join-Path $env:VSINSTALLDIR 'Common7\IDE\CommonExtensions\Microsoft\CMake'
    foreach ($relative in @('CMake\bin', 'Ninja')) {
        $directory = Join-Path $bundled $relative
        if (Test-Path $directory) { $env:Path += ';' + $directory }
    }
}
if (-not (Get-Command cmake.exe -ErrorAction SilentlyContinue)) { throw 'CMake 3.28+ is required.' }

if ($Backend -eq 'cuda') {
    if (-not $env:CUDA_PATH -or -not (Test-Path (Join-Path $env:CUDA_PATH 'bin\nvcc.exe'))) {
        throw 'Install the CUDA Toolkit and set CUDA_PATH before running this job.'
    }
    $env:Path = (Join-Path $env:CUDA_PATH 'bin') + ';' + $env:Path
}

if ($env:PYTHON_EXECUTABLE) {
    if (-not (Test-Path $env:PYTHON_EXECUTABLE -PathType Leaf)) { throw 'PYTHON_EXECUTABLE does not identify a file.' }
    $pythonPath = (Resolve-Path $env:PYTHON_EXECUTABLE).Path
} else {
    $python = Get-Command python.exe,python3.exe,py.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.Source -notlike '*\Microsoft\WindowsApps\*' } | Select-Object -First 1
    if (-not $python) { throw 'Python 3.10+ is required; install it or set PYTHON_EXECUTABLE for the runner.' }
    $pythonPath = $python.Source
}
$prefix = @()
if ([IO.Path]::GetFileName($pythonPath) -eq 'py.exe') { $prefix = @('-3') }
$env:Path = (Split-Path $pythonPath) + ';' + $env:Path
& $pythonPath @prefix -c 'import sys; sys.exit(int(sys.version_info < (3, 10)))'
if ($LASTEXITCODE -ne 0) { throw 'Python validation failed.' }
& $pythonPath @prefix ci/gitlab/build_runtime.py --backend $Backend
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
