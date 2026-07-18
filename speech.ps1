param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $SpeechArgs
)

$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$VenvPythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$PortableEnv = @{
    SPEECH_HOME = $Root
    PYTHONPATH = $Root
    SPEECH_DATA_DIR = (Join-Path $Root "data")
    HF_HOME = (Join-Path $Root "models\huggingface")
    HF_HUB_CACHE = (Join-Path $Root "models\huggingface\hub")
    TRANSFORMERS_CACHE = (Join-Path $Root "models\huggingface\transformers")
    TORCH_HOME = (Join-Path $Root "models\torch")
    XDG_CACHE_HOME = (Join-Path $Root "cache")
    PIP_CACHE_DIR = (Join-Path $Root "cache\pip")
    TMP = (Join-Path $Root "tmp")
    TEMP = (Join-Path $Root "tmp")
}

foreach ($entry in $PortableEnv.GetEnumerator()) {
    [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
}

foreach ($path in @(
    $env:SPEECH_DATA_DIR,
    $env:HF_HOME,
    $env:HF_HUB_CACHE,
    $env:TRANSFORMERS_CACHE,
    $env:TORCH_HOME,
    $env:XDG_CACHE_HOME,
    $env:PIP_CACHE_DIR,
    $env:TMP
)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

function Get-BasePython {
    if (Test-Path -LiteralPath $VenvPython) {
        return $VenvPython
    }
    return "python"
}

function Get-AppPython {
    if (Test-Path -LiteralPath $VenvPythonw) {
        return $VenvPythonw
    }
    return Get-BasePython
}

function New-SpeechVenv {
    if (Test-Path -LiteralPath $VenvPython) {
        return
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv (Join-Path $Root ".venv")
        if (Test-Path -LiteralPath $VenvPython) {
            return
        }
        Write-Host "Python 3.11 launcher is unavailable; trying installed python."
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        & python -m venv (Join-Path $Root ".venv")
        if (Test-Path -LiteralPath $VenvPython) {
            return
        }
        Write-Host "Installed python could not create a venv; trying winget."
    }

    Install-PythonWithWinget
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        & py -3.11 -m venv (Join-Path $Root ".venv")
        return
    }
    & python -m venv (Join-Path $Root ".venv")
}

function Install-PythonWithWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Python 3.11 is required. Install it from https://www.python.org/downloads/ and run install again."
    }

    Write-Host "Installing Python 3.11 with winget..."
    & winget install --id Python.Python.3.11 --exact --scope user --accept-package-agreements --accept-source-agreements
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath) {
        $env:Path = $env:Path + ";" + $userPath
    }
}

function Install-Speech {
    Write-Host "Creating Speech environment..."
    New-SpeechVenv

    Write-Host "Installing Speech runtime dependencies..."
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $Root "requirements.txt")

    Write-Host "Installing Parakeet dependencies..."
    & $VenvPython -m pip install -r (Join-Path $Root "requirements-parakeet.txt")

    Write-Host "Installing Whisper (faster-whisper) dependencies..."
    & $VenvPython -m pip install -r (Join-Path $Root "requirements-whisper.txt")

    $binDir = Join-Path $Root "bin"
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    $pathParts = @()
    if ($userPath) {
        $pathParts = $userPath -split ";" | Where-Object { $_ }
    }
    $alreadyOnPath = $pathParts | Where-Object {
        $_.TrimEnd("\") -ieq $binDir.TrimEnd("\")
    }
    if (-not $alreadyOnPath) {
        $newUserPath = (($pathParts + $binDir) -join ";")
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        $env:Path = $env:Path + ";" + $binDir
        Write-Host "Added $binDir to User PATH."
    }

    Write-Host ""
    Write-Host "Speech install complete."
    Write-Host "Run: speech"
    Write-Host "Download Parakeet: speech parakeet install"
}

function Invoke-SpeechPython {
    param([string[]] $ArgsForPython)
    $python = Get-BasePython
    Set-Location -LiteralPath $Root
    & $python -m speech_app @ArgsForPython
}

function Start-SpeechDetached {
    param([switch] $ShowWindow)
    $python = Get-AppPython
    $arguments = @("-m", "speech_app", "run")
    if ($ShowWindow) {
        $arguments += "--show-window"
    }
    Set-Location -LiteralPath $Root
    Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory $Root `
        -WindowStyle Hidden
    Write-Host "Speech started in tray."
}

function Start-SpeechUi {
    $releaseExe = Join-Path $Root "tauri\src-tauri\target\release\speech-tauri.exe"
    if (Test-Path -LiteralPath $releaseExe) {
        Start-Process `
            -FilePath $releaseExe `
            -WorkingDirectory $Root
        Write-Host "Speech UI opened."
        return
    }

    $tauriDir = Join-Path $Root "tauri"
    $packageJson = Join-Path $tauriDir "package.json"
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npm) {
        $npm = Get-Command npm -ErrorAction SilentlyContinue
    }

    if ($npm -and (Test-Path -LiteralPath $packageJson)) {
        Start-Process `
            -FilePath $npm.Source `
            -ArgumentList @("run", "tauri:dev") `
            -WorkingDirectory $tauriDir `
            -WindowStyle Hidden
        Write-Host "Speech UI opened in Tauri dev mode."
        return
    }

    Write-Host "Tauri UI is unavailable; opening fallback window."
    Stop-SpeechProcesses
    Start-SpeechDetached -ShowWindow
}

function Get-SpeechProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*speech_app*" -and
            $_.CommandLine -like "*$Root*"
        }
}

function Stop-SpeechProcesses {
    $processes = @(Get-SpeechProcesses)
    if ($processes.Count -eq 0) {
        Write-Host "Speech is not running."
        return
    }

    foreach ($process in $processes) {
        Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
    Write-Host "Speech stopped."
}

function Show-SpeechStatus {
    $processes = @(Get-SpeechProcesses)
    if ($processes.Count -eq 0) {
        Write-Host "Speech is not running."
        return
    }

    $processes | Select-Object ProcessId, Name, CommandLine | Format-Table -AutoSize
}

function Ensure-SpeechRunning {
    $processes = @(Get-SpeechProcesses)
    if ($processes.Count -eq 0) {
        Start-SpeechDetached
    }
}

if ($SpeechArgs.Count -eq 0) {
    Start-SpeechDetached
    exit 0
}

$command = $SpeechArgs[0].ToLowerInvariant()
$tail = @()
if ($SpeechArgs.Count -gt 1) {
    $tail = $SpeechArgs[1..($SpeechArgs.Count - 1)]
}

switch ($command) {
    "install" {
        Install-Speech
        exit 0
    }
    "run" {
        Start-SpeechDetached
        exit 0
    }
    "start" {
        Start-SpeechDetached
        exit 0
    }
    "stop" {
        Stop-SpeechProcesses
        exit 0
    }
    "restart" {
        Stop-SpeechProcesses
        Start-SpeechDetached
        exit 0
    }
    "open" {
        Ensure-SpeechRunning
        Start-SpeechUi
        exit 0
    }
    "status" {
        Show-SpeechStatus
        exit 0
    }
    "foreground" {
        Invoke-SpeechPython @("run")
        exit $LASTEXITCODE
    }
    "diagnose" {
        Invoke-SpeechPython @("diagnose")
        exit $LASTEXITCODE
    }
    "parakeet" {
        if ($tail.Count -eq 0) {
            Write-Host "Usage: speech parakeet install"
            exit 1
        }
        $argsForPython = @("parakeet") + $tail
        Invoke-SpeechPython $argsForPython
        exit $LASTEXITCODE
    }
    "model" {
        if ($tail.Count -eq 0) {
            Write-Host "Usage: speech model install [parakeet|whisper-ru]"
            Write-Host "       speech model list"
            exit 1
        }
        # Map legacy "model install/preload/download" (no preset arg) to the
        # active model; otherwise pass through to the speech_app model parser.
        if ($tail.Count -eq 1 -and $tail[0].ToLowerInvariant() -in @("install", "preload", "download")) {
            Invoke-SpeechPython @("model", "install")
        } else {
            $argsForPython = @("model") + $tail
            Invoke-SpeechPython $argsForPython
        }
        exit $LASTEXITCODE
    }
    default {
        Invoke-SpeechPython $SpeechArgs
        exit $LASTEXITCODE
    }
}
