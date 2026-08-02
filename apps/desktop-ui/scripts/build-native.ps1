$ErrorActionPreference = 'Stop'

$desktopRoot = Split-Path -Parent $PSScriptRoot
$sensRoot = (Resolve-Path (Join-Path $desktopRoot '..\..')).Path
$packageTarget = Join-Path $sensRoot 'target\package'
$releaseRoot = Join-Path $packageTarget 'release'
$env:CARGO_TARGET_DIR = $packageTarget
$tauriConfigPath = Join-Path $desktopRoot 'src-tauri\tauri.conf.json'
$appVersion = (Get-Content -Raw -LiteralPath $tauriConfigPath | ConvertFrom-Json).version

if ([string]::IsNullOrWhiteSpace($env:TAURI_SIGNING_PRIVATE_KEY)) {
    $userProfile = [Environment]::GetFolderPath('UserProfile')
    $localSigningKey = Join-Path $userProfile '.sens-release\sens-updater.key'
    if (-not (Test-Path -LiteralPath $localSigningKey -PathType Leaf)) {
        throw "Sens updater signing key is missing. Expected $localSigningKey or TAURI_SIGNING_PRIVATE_KEY_PATH."
    }
    $env:TAURI_SIGNING_PRIVATE_KEY = $localSigningKey
}

if ([string]::IsNullOrWhiteSpace($env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD)) {
    $userProfile = [Environment]::GetFolderPath('UserProfile')
    $passwordPath = Join-Path $userProfile '.sens-release\sens-updater.password.dpapi'
    if (-not (Test-Path -LiteralPath $passwordPath -PathType Leaf)) {
        throw "Sens updater key password is missing. Expected $passwordPath or TAURI_SIGNING_PRIVATE_KEY_PASSWORD."
    }
    $securePassword = (Get-Content -Raw -LiteralPath $passwordPath).Trim() | ConvertTo-SecureString
    $credential = New-Object System.Management.Automation.PSCredential('sens-updater', $securePassword)
    $env:TAURI_SIGNING_PRIVATE_KEY_PASSWORD = $credential.GetNetworkCredential().Password
}

& cargo build --manifest-path (Join-Path $sensRoot 'Cargo.toml') --release -p sens-broker -p sens-mcp -p sens-connect
if ($LASTEXITCODE -ne 0) {
    throw "Sidecar build failed with exit code $LASTEXITCODE"
}

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'prepare-native-sidecars.ps1') -TargetRoot $releaseRoot
if ($LASTEXITCODE -ne 0) {
    throw "Sidecar preparation failed with exit code $LASTEXITCODE"
}

Push-Location $desktopRoot
try {
    & npx tauri build --config src-tauri/tauri.bundle.conf.json
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

$canonicalBundle = Join-Path $sensRoot 'target\release\bundle'
$nsisName = "Sens_${appVersion}_x64-setup.exe"
$msiName = "Sens_${appVersion}_x64_ru-RU.msi"
$artifacts = @(
    @{
        Source = Join-Path $releaseRoot "bundle\nsis\$nsisName"
        Destination = Join-Path $canonicalBundle "nsis\$nsisName"
    },
    @{
        Source = Join-Path $releaseRoot "bundle\nsis\$nsisName.sig"
        Destination = Join-Path $canonicalBundle "nsis\$nsisName.sig"
    },
    @{
        Source = Join-Path $releaseRoot "bundle\msi\$msiName"
        Destination = Join-Path $canonicalBundle "msi\$msiName"
    },
    @{
        Source = Join-Path $releaseRoot "bundle\msi\$msiName.sig"
        Destination = Join-Path $canonicalBundle "msi\$msiName.sig"
    }
)

foreach ($artifact in $artifacts) {
    if (-not (Test-Path -LiteralPath $artifact.Source -PathType Leaf)) {
        throw "Missing installer: $($artifact.Source)"
    }
    $destinationDirectory = Split-Path -Parent $artifact.Destination
    New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null
    Copy-Item -LiteralPath $artifact.Source -Destination $artifact.Destination -Force
}

$signature = (Get-Content -Raw -LiteralPath (Join-Path $canonicalBundle "nsis\$nsisName.sig")).Trim()
$releaseBase = "https://github.com/TheRofli/Sens/releases/download/v$appVersion"
$manifest = [ordered]@{
    version = $appVersion
    notes = "Sens ${appVersion}: managed dictation, custom tray menu, movable window, hidden background processes, and in-app updates."
    pub_date = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    platforms = [ordered]@{
        'windows-x86_64' = [ordered]@{
            signature = $signature
            url = "$releaseBase/$nsisName"
        }
    }
}
$manifestPath = Join-Path $canonicalBundle 'latest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$artifacts | ForEach-Object {
    Get-Item -LiteralPath $_.Destination | Select-Object FullName, Length, LastWriteTime
}
Get-Item -LiteralPath $manifestPath | Select-Object FullName, Length, LastWriteTime
