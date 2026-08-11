param(
    [Parameter(Mandatory = $true)]
    [string]$SourceUrl,
    [string]$CandidateUrl,
    [string]$McpPath,
    [string]$AssetOutputDir,
    [int]$Width = 800,
    [int]$Height = 600,
    [switch]$UseSourceSidecars
)

$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($McpPath)) {
    $McpPath = Join-Path $PSScriptRoot '..\target\debug\sens-mcp.exe'
}
if ([string]::IsNullOrWhiteSpace($AssetOutputDir)) {
    $AssetOutputDir = Join-Path $PSScriptRoot '..\output\mcp-session-smoke\assets'
}
$McpPath = [System.IO.Path]::GetFullPath($McpPath)
$AssetOutputDir = [System.IO.Path]::GetFullPath($AssetOutputDir)
[System.IO.Directory]::CreateDirectory($AssetOutputDir) | Out-Null

if (-not [System.IO.File]::Exists($McpPath)) {
    throw "sens-mcp executable was not found: $McpPath"
}

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $McpPath
$startInfo.WorkingDirectory = [System.IO.Path]::GetDirectoryName($McpPath)
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$startInfo.CreateNoWindow = $true
if ($UseSourceSidecars) {
    $sourceSidecars = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\sidecars'))
    $startInfo.EnvironmentVariables['SENS_SIDECARS_ROOT'] = $sourceSidecars
}
$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
[void]$process.Start()

function Invoke-McpRequest {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Method,
        [hashtable]$Params = @{}
    )
    $message = @{
        jsonrpc = '2.0'
        id = $Id
        method = $Method
        params = $Params
    } | ConvertTo-Json -Depth 30 -Compress
    $process.StandardInput.WriteLine($message)
    $process.StandardInput.Flush()
    while ($true) {
        $line = $process.StandardOutput.ReadLine()
        if (-not $line) {
            $stderr = $process.StandardError.ReadToEnd()
            throw "sens-mcp closed before response $Id. $stderr"
        }
        $response = $line | ConvertFrom-Json
        if ($response.id -eq $Id) {
            if ($response.error) {
                throw ($response.error | ConvertTo-Json -Depth 10 -Compress)
            }
            return $response.result
        }
    }
}

function Invoke-SensTool {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Id,
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [hashtable]$Arguments
    )
    $result = Invoke-McpRequest -Id $Id -Method 'tools/call' -Params @{
        name = $Name
        arguments = $Arguments
    }
    if ($result.isError) {
        throw ($result.content | ConvertTo-Json -Depth 10 -Compress)
    }
    $envelope = $result.structuredContent.result
    if ($envelope.status -ne 'succeeded') {
        throw ($envelope.error | ConvertTo-Json -Depth 10 -Compress)
    }
    return $envelope.data
}

try {
    $null = Invoke-McpRequest -Id 1 -Method 'initialize' -Params @{
        protocolVersion = '2025-06-18'
        capabilities = @{}
        clientInfo = @{ name = 'sens-web-session-smoke'; version = '1.0' }
    }
    $process.StandardInput.WriteLine((@{
        jsonrpc = '2.0'
        method = 'notifications/initialized'
        params = @{}
    } | ConvertTo-Json -Depth 10 -Compress))
    $process.StandardInput.Flush()

    $tools = Invoke-McpRequest -Id 2 -Method 'tools/list'
    $toolNames = @($tools.tools | ForEach-Object { $_.name })
    foreach ($required in @('sens_web_start', 'sens_web_review')) {
        if ($required -notin $toolNames) {
            throw "MCP tool is missing: $required"
        }
    }

    $start = Invoke-SensTool -Id 3 -Name 'sens_web_start' -Arguments @{
        sourceUrl = $SourceUrl
        prompt = 'Reconstruct this source as a real website with live selectable text and semantic controls.'
        assetOutputDir = $AssetOutputDir
        viewport = @{ width = $Width; height = $Height }
        dpr = 1.0
        theme = 'light'
        locale = 'en-US'
        waitUntil = 'load'
        timeoutMs = 30000
        settleMs = 0
        fast = $true
        maxCalls = 0
    }
    $sessionId = [string]$start.webSession.sessionId
    if (-not $sessionId) {
        throw 'sens_web_start did not return a sessionId'
    }

    $summary = [ordered]@{
        sessionId = $sessionId
        sourceSha256 = $start.sourceCapture.screenshotSha256
        sourceFrozen = $start.webSession.sourceFrozen
        contractPath = $start.contractPath
        startRequiredAction = $start.requiredAction
        reviewCount = 0
        beforeAfterLinked = $false
        finalReceiptIssued = $false
        finalCanComplete = $false
    }

    if ($CandidateUrl) {
        $review = Invoke-SensTool -Id 4 -Name 'sens_web_review' -Arguments @{
            sessionId = $sessionId
            candidateUrl = $CandidateUrl
            final = $false
        }
        $finalReview = Invoke-SensTool -Id 5 -Name 'sens_web_review' -Arguments @{
            sessionId = $sessionId
            final = $true
        }
        $summary.reviewCount = $finalReview.webSession.reviewCount
        $summary.firstReviewReport = $review.reviewReport.path
        $summary.finalReviewReport = $finalReview.reviewReport.path
        $summary.reviewReportsPersisted = (
            (Test-Path -LiteralPath ([string]$review.reviewReport.path)) -and
            (Test-Path -LiteralPath ([string]$finalReview.reviewReport.path))
        )
        $summary.beforeAfterLinked = (
            $finalReview.beforeCapture.sha256 -eq $review.afterCapture.sha256 -and
            -not [string]::IsNullOrWhiteSpace([string]$review.afterCapture.sha256)
        )
        $summary.finalReceiptIssued = $null -ne $finalReview.completionReceipt
        $summary.finalCanComplete = [bool]$finalReview.canComplete
        $summary.finalRequiredAction = $finalReview.requiredAction
        $summary.visualPass = [bool]$finalReview.visualPass
        $summary.webPass = [bool]$finalReview.webPass
    }

    $summaryJson = [pscustomobject]$summary | ConvertTo-Json -Depth 10
    [Console]::Out.WriteLine($summaryJson)
}
finally {
    try {
        $process.StandardInput.Close()
    }
    catch {
    }
    if (-not $process.WaitForExit(3000)) {
        $process.Kill()
    }
    $process.Dispose()
}
