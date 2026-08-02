param(
    [string]$TargetRoot
)

$ErrorActionPreference = 'Stop'

$desktopRoot = Split-Path -Parent $PSScriptRoot
$sensRoot = (Resolve-Path (Join-Path $desktopRoot '..\..')).Path
if ([string]::IsNullOrWhiteSpace($TargetRoot)) {
    $cargoTarget = if ([string]::IsNullOrWhiteSpace($env:CARGO_TARGET_DIR)) {
        Join-Path $sensRoot 'target'
    } else {
        $env:CARGO_TARGET_DIR
    }
    $TargetRoot = Join-Path $cargoTarget 'release'
}
$targetRoot = [System.IO.Path]::GetFullPath($TargetRoot)
$binaryRoot = Join-Path $desktopRoot 'src-tauri\binaries'

New-Item -ItemType Directory -Force -Path $binaryRoot | Out-Null

$targetTriple = 'x86_64-pc-windows-msvc'
$names = @('sens-broker', 'sens-mcp', 'sens-connect')
foreach ($name in $names) {
    $source = Join-Path $targetRoot "$name.exe"
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing release binary: $source"
    }
    $destination = Join-Path $binaryRoot "$name-$targetTriple.exe"
    Copy-Item -LiteralPath $source -Destination $destination -Force
}

Get-ChildItem -LiteralPath $binaryRoot -File | Select-Object Name, Length
