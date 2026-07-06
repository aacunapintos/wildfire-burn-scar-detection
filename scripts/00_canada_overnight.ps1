# 00_canada_overnight.ps1
# Download + Patches + Drive sync for North Slave Complex (Canada boreal, Aug 2023).
#
# Launch:  powershell -File D:\GeoAI\wildfire-spread\scripts\00_canada_overnight.ps1
#
# Monitor from a second terminal (real-time):
#   Get-Content D:\GeoAI\wildfire-spread\data\canada_overnight.log -Wait -Tail 40

$ProjectRoot = "D:\GeoAI\wildfire-spread"
$LogFile     = "$ProjectRoot\data\canada_overnight.log"
$PatchSrc    = "$ProjectRoot\data\canada\patches"
$PatchDst    = "G:\Mon Drive\GeoAI\wildfire-spread\data\canada\patches"

New-Item -ItemType Directory -Force -Path "$ProjectRoot\data" | Out-Null
"" | Set-Content -Path $LogFile -Encoding utf8

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

Log "===== Canada overnight pipeline started ====="
Log "Project : $ProjectRoot"
Log "Log     : $LogFile"
Log "Monitor : Get-Content $LogFile -Wait -Tail 40"
Log ""

# ── Step 1: Download + Patches (single script) ────────────────────────────────
Log "STEP 1 — Download + Patches (11_canada_pipeline.py)"
Log "Estimated: 6-10 hours depending on scenes found"
Log ""

# Tee-Object: output goes to terminal AND log file simultaneously.
# python -u ensures stdout is unbuffered (no delay between lines).
$startTime = Get-Date

& python -u "$ProjectRoot\scripts\11_canada_pipeline.py" 2>&1 | Tee-Object -FilePath $LogFile -Append

$exitCode = $LASTEXITCODE

Log ""
$elapsed = [math]::Round(((Get-Date) - $startTime).TotalMinutes, 1)
Log "Pipeline exit code : $exitCode  (elapsed: ${elapsed} min)"

if ($exitCode -ne 0) {
    Log "PIPELINE FAILED — aborting Drive sync."
    Log "Scroll up to find the ERROR line printed by python."
    exit 1
}

$npy_img  = (Get-ChildItem "$PatchSrc\images"     -Filter "*.npy" -ErrorAction SilentlyContinue).Count
$npy_mask = (Get-ChildItem "$PatchSrc\masks_dnbr" -Filter "*.npy" -ErrorAction SilentlyContinue).Count
$jp2count = (Get-ChildItem "$ProjectRoot\data\canada\raw_prefire"  -Recurse -Filter "*.jp2" -ErrorAction SilentlyContinue).Count +
            (Get-ChildItem "$ProjectRoot\data\canada\raw_postfire" -Recurse -Filter "*.jp2" -ErrorAction SilentlyContinue).Count
$dlGB     = [math]::Round(((Get-ChildItem "$ProjectRoot\data\canada" -Recurse -Filter "*.jp2" -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum) / 1GB, 2)

Log "JP2 files   : $jp2count  ($dlGB GB)"
Log "Patches img : $npy_img"
Log "Patches msk : $npy_mask"
Log ""

# ── Step 2: Drive sync ────────────────────────────────────────────────────────
Log "STEP 2 — Syncing patches to Google Drive"
Log "Destination: $PatchDst"

New-Item -ItemType Directory -Force -Path $PatchDst | Out-Null

robocopy $PatchSrc $PatchDst /E /MT:4 /NP /NFL /NDL /LOG+:$LogFile

$rcExit = $LASTEXITCODE
if ($rcExit -ge 8) {
    Log "ROBOCOPY FAILED (exit $rcExit) — check log."
    exit 1
}

Log "Drive sync complete."
Log ""

# ── Summary ───────────────────────────────────────────────────────────────────
Log "===== ALL DONE ====="
Log "  JP2 files   : $jp2count  ($dlGB GB)"
Log "  Patches     : $npy_img  images / $npy_mask masks"
Log "  Drive path  : $PatchDst"
Log ""
Log "Next step: open Colab A100 and run notebooks/10_canada_zs_evaluation.ipynb"
