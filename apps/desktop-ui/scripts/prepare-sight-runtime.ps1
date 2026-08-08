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
$sightRequirements = Join-Path $sensRoot 'sidecars\sight\requirements-runtime.txt'
$hearingRequirements = Join-Path $sensRoot 'sidecars\speech\requirements-runtime.txt'
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
    throw "Sens runtime packaging requires a Python 3.11 build interpreter"
}
$builderRoot = (& $PythonCommand -c "import sys; print(sys.base_prefix)").Trim()
if (-not (Test-Path -LiteralPath (Join-Path $builderRoot 'Lib\tkinter') -PathType Container)) {
    throw "The Python 3.11 build interpreter does not include Tk"
}

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null
Expand-Archive -LiteralPath $archivePath -DestinationPath $stagingRoot -Force

# Embeddable Python omits Tk. Hearing keeps the proven Ctrl+Win overlay and
# hotkey flow, so package only the matching Python 3.11 Tk components.
New-Item -ItemType Directory -Force -Path (Join-Path $stagingRoot 'Lib') | Out-Null
Copy-Item -LiteralPath (Join-Path $builderRoot 'Lib\tkinter') -Destination (Join-Path $stagingRoot 'Lib\tkinter') -Recurse -Force
Copy-Item -LiteralPath (Join-Path $builderRoot 'tcl') -Destination (Join-Path $stagingRoot 'tcl') -Recurse -Force
foreach ($nativeTkFile in @('_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll')) {
    $source = Join-Path $builderRoot "DLLs\$nativeTkFile"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing Tk runtime component: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $stagingRoot $nativeTkFile) -Force
}

$sitePackages = Join-Path $stagingRoot 'Lib\site-packages'
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null
& $PythonCommand -m pip install `
    --disable-pip-version-check `
    --ignore-installed `
    --no-compile `
    --extra-index-url 'https://abetlen.github.io/llama-cpp-python/whl/cpu' `
    --target $sitePackages `
    --requirement $sightRequirements `
    --requirement $hearingRequirements
if ($LASTEXITCODE -ne 0) {
    throw "Sens Sight + Hearing Python dependencies failed to resolve together"
}

$pthPath = Join-Path $stagingRoot 'python311._pth'
$pth = Get-Content -Raw -LiteralPath $pthPath
$pth = $pth -replace '#import site', 'import site'
if ($pth -notmatch '(?m)^Lib\\site-packages$') {
    $pth = $pth.TrimEnd() + "`r`nLib\site-packages`r`n"
}
if ($pth -notmatch '(?m)^Lib$') {
    $pth = $pth.TrimEnd() + "`r`nLib`r`n"
}
[System.IO.File]::WriteAllText($pthPath, $pth, [System.Text.UTF8Encoding]::new($false))

$manifest = [ordered]@{
    schemaVersion = 1
    python = $pythonVersion
    architecture = 'windows-x86_64'
    device = 'cpu'
    sightRequirementsSha256 = (Get-Sha256Hex -LiteralPath $sightRequirements).ToLowerInvariant()
    hearingRequirementsSha256 = (Get-Sha256Hex -LiteralPath $hearingRequirements).ToLowerInvariant()
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

$tclLibrary = Join-Path $stagingRoot 'tcl\tcl8.6'
$tkLibrary = Join-Path $stagingRoot 'tcl\tk8.6'
$smoke = "import os, sys; os.environ.pop('CUDA_PATH', None); os.environ.pop('HIP_PATH', None); os.environ['TCL_LIBRARY'] = sys.argv[1]; os.environ['TK_LIBRARY'] = sys.argv[2]; import cv2, faster_whisper, llama_cpp, numpy, onnxruntime, playwright, psutil, pynput, pyperclip, pystray, rapidocr, sherpa_onnx, sounddevice, tkinter, yt_dlp; tkinter.Tcl(); print('Sens Sight + Hearing runtime OK')"
& (Join-Path $stagingRoot 'python.exe') -I -c $smoke $tclLibrary $tkLibrary
if ($LASTEXITCODE -ne 0) {
    throw "Packaged Sens runtime failed its isolated import smoke test"
}

if (Test-Path -LiteralPath $runtimeRoot) {
    Remove-Item -LiteralPath $runtimeRoot -Recurse -Force
}
Move-Item -LiteralPath $stagingRoot -Destination $runtimeRoot
Get-ChildItem -LiteralPath $runtimeRoot -File | Select-Object Name, Length
