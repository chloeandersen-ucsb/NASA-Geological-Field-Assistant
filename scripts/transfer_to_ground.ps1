param(
    [string]$JetsonHost = "",
    [string]$JetsonUser = "",
    [string]$DestRoot = "",
    [int]$Port = 22,
    [switch]$SkipConnectionTest
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found. Install/enable OpenSSH client on Windows."
    }
}

function Ensure-Dir {
    param([string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Resolve-ConnectionInfo {
    if ([string]::IsNullOrWhiteSpace($JetsonUser)) {
        Set-Variable -Scope Script -Name JetsonUser -Value (Read-Host "Jetson username (run 'whoami' on Jetson)")
    }

    if ([string]::IsNullOrWhiteSpace($JetsonHost)) {
        Set-Variable -Scope Script -Name JetsonHost -Value (Read-Host "Jetson IP/hostname (run 'hostname -I' on Jetson)")
    }

    if ([string]::IsNullOrWhiteSpace($JetsonUser) -or [string]::IsNullOrWhiteSpace($JetsonHost)) {
        throw "Jetson username and host are required."
    }
}

function Resolve-DestinationRoot {
    if ([string]::IsNullOrWhiteSpace($DestRoot)) {
        $Downloads = Join-Path ([Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)) "Downloads"
        Set-Variable -Scope Script -Name DestRoot -Value (Join-Path $Downloads "sage_ground_data")
    }
}

function Resolve-RunRoot {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    Set-Variable -Scope Script -Name RunRoot -Value (Join-Path $DestRoot $timestamp)
}

function Test-Connection {
    $target = "${JetsonUser}@${JetsonHost}"
    Write-Host "[CHECK] Testing SSH to ${target}:${Port}"
    & ssh -p $Port $target "echo connected"
    if ($LASTEXITCODE -ne 0) {
        throw "SSH connectivity test failed for $target on port $Port."
    }
}

function Copy-RemoteDir {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )

    Ensure-Dir -Path $LocalPath
    $remoteSpec = "${JetsonUser}@${JetsonHost}:${RemotePath}/*"
    Write-Host "[COPY] $remoteSpec -> $LocalPath"

    & scp -P $Port -r $remoteSpec $LocalPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Skipped (missing/inaccessible): $RemotePath"
    }
}

function Copy-RemoteFile {
    param(
        [string]$RemotePath,
        [string]$LocalPath
    )

    Ensure-Dir -Path (Split-Path -Parent $LocalPath)
    $remoteSpec = "${JetsonUser}@${JetsonHost}:${RemotePath}"
    Write-Host "[COPY] $remoteSpec -> $LocalPath"

    & scp -P $Port $remoteSpec $LocalPath
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Skipped (missing/inaccessible): $RemotePath"
    }
}

Require-Command -Name scp
Require-Command -Name ssh
Resolve-ConnectionInfo
Resolve-DestinationRoot
Resolve-RunRoot

if (-not $SkipConnectionTest) {
    Test-Connection
}

$imagesDir = Join-Path $RunRoot "images"
$voiceDir = Join-Path $RunRoot "voice_notes"
$sageRepoDir = Join-Path $RunRoot "sage_repo"
$sageStoreDir = Join-Path $RunRoot "sage_store"
$logsDir = Join-Path $RunRoot "logs"

Ensure-Dir -Path $imagesDir
Ensure-Dir -Path $voiceDir
Ensure-Dir -Path $sageRepoDir
Ensure-Dir -Path $sageStoreDir
Ensure-Dir -Path $logsDir

Copy-RemoteDir -RemotePath "/home/jetson/CapstoneGit/capstone/ML-classifications/camera-pipeline/images" -LocalPath $imagesDir
Copy-RemoteDir -RemotePath "/home/jetson/CapstoneGit/capstone/voiceNotes/data" -LocalPath $voiceDir
Copy-RemoteDir -RemotePath "/home/jetson/CapstoneGit/capstone/led-display/sage_data" -LocalPath $sageRepoDir
Copy-RemoteDir -RemotePath "/data/sage" -LocalPath $sageStoreDir
Copy-RemoteFile -RemotePath "/home/jetson/CapstoneGit/capstone/log.txt" -LocalPath (Join-Path $logsDir "log.txt")

Write-Host "Done. Output files pulled to: $RunRoot"
