param(
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = 'Stop'

function Get-Sha256Hex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )

    $stream = [System.IO.File]::OpenRead($LiteralPath)
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '')
    }
    finally {
        $sha256.Dispose()
        $stream.Dispose()
    }
}

$desktopRoot = Split-Path -Parent $PSScriptRoot
$sensRoot = (Resolve-Path (Join-Path $desktopRoot '..\..')).Path
$runtimeRoot = [System.IO.Path]::GetFullPath((Join-Path $desktopRoot 'src-tauri\runtime\python'))
$allowedRoot = [System.IO.Path]::GetFullPath((Join-Path $desktopRoot 'src-tauri\runtime'))
if (-not $runtimeRoot.StartsWith($allowedRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to prepare a runtime outside $allowedRoot"
}

$pythonVersion = '3.11.9'
$archiveName = "python-$pythonVersion-embed-amd64.zip"
$archiveUrl = "https://www.python.org/ftp/python/$pythonVersion/$archiveName"
$archiveSha256 = '009D6BF7E3B2DDCA3D784FA09F90FE54336D5B60F0E0F305C37F400BF83CFD3B'
$requirements = Join-Path $sensRoot 'sidecars\sight\requirements-runtime.txt'
$downloadRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'sens-runtime-downloads'
$archivePath = Join-Path $downloadRoot $archiveName
$stagingRoot = Join-Path $allowedRoot 'python.staging'

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf) -or
    (Get-Sha256Hex -LiteralPath $archivePath) -ne $archiveSha256) {
    Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -UseBasicParsing -Uri $archiveUrl -OutFile $archivePath
}
if ((Get-Sha256Hex -LiteralPath $archivePath) -ne $archiveSha256) {
    throw "Python runtime archive failed SHA-256 verification"
}

$builderVersion = & $PythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $builderVersion.Trim() -ne '3.11') {
    throw "Sight runtime packaging requires a Python 3.11 build interpreter"
}

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingRoot -Force

$sitePackages = Join-Path $stagingRoot 'Lib\site-packages'
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null
& $PythonCommand -m pip install `
    --disable-pip-version-check `
    --ignore-installed `
    --no-compile `
    --extra-index-url 'https://abetlen.github.io/llama-cpp-python/whl/cpu' `
    --target $sitePackages `
    --requirement $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Sight Python dependencies failed to install"
}

$pthPath = Join-Path $stagingRoot 'python311._pth'
$pth = Get-Content -Raw -LiteralPath $pthPath
$pth = $pth -replace '#import site', 'import site'
if ($pth -notmatch '(?m)^Lib\\site-packages$') {
    $pth = $pth.TrimEnd() + "`r`nLib\site-packages`r`n"
}
[System.IO.File]::WriteAllText($pthPath, $pth, [System.Text.UTF8Encoding]::new($false))

$manifest = [ordered]@{
    schemaVersion = 1
    python = $pythonVersion
    architecture = 'windows-x86_64'
    device = 'cpu'
    requirementsSha256 = (Get-Sha256Hex -LiteralPath $requirements).ToLowerInvariant()
}
$manifestJson = $manifest | ConvertTo-Json -Depth 3
[System.IO.File]::WriteAllText(
    (Join-Path $stagingRoot 'sens-runtime.json'),
    $manifestJson,
    [System.Text.UTF8Encoding]::new($false)
)
New-Item -ItemType File -Force -Path (Join-Path $stagingRoot '.gitkeep') | Out-Null

$sidecarsRoot = Join-Path $sensRoot 'sidecars'
$ocrSmoke = "import sys; sys.path.insert(0, sys.argv[1]); from sight.ocr import ocr_engine; ocr_engine(); print('Sight OCR models OK')"
& (Join-Path $stagingRoot 'python.exe') -I -c $ocrSmoke $sidecarsRoot
if ($LASTEXITCODE -ne 0) {
    throw "Packaged Sight runtime failed to preload its OCR models"
}

$smoke = "import os; os.environ.pop('CUDA_PATH', None); os.environ.pop('HIP_PATH', None); import cv2, llama_cpp, numpy, onnxruntime, playwright, rapidocr; print('Sight runtime OK')"
& (Join-Path $stagingRoot 'python.exe') -I -c $smoke
if ($LASTEXITCODE -ne 0) {
    throw "Packaged Sight runtime failed its isolated import smoke test"
}

if (Test-Path -LiteralPath $runtimeRoot) {
    Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
}
Move-Item -LiteralPath $stagingRoot -Destination $runtimeRoot
Get-ChildItem -LiteralPath $runtimeRoot -File | Select-Object Name, Length
