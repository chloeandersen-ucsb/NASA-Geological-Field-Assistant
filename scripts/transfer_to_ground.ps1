param(
    [string]$JetsonHost = "192.168.0.22",
    [string]$JetsonUser = "jetson",
    [string]$DestRoot = "C:\Users\Bruce\ground_data",
    [int]$Port = 22
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

$imagesDir = Join-Path $DestRoot "images"
$voiceDir = Join-Path $DestRoot "voice_notes"
$sageRepoDir = Join-Path $DestRoot "sage_repo"
$sageStoreDir = Join-Path $DestRoot "sage_store"
$logsDir = Join-Path $DestRoot "logs"

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

Write-Host "Done. Output files pulled to: $DestRoot"
