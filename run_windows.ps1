# run_windows.ps1 — uruchamianie projektu na Windows bez WSL

param(

    [switch]$Verify,

    [switch]$PrepareData,

    [switch]$TrainAll,

    [switch]$Evaluate,

    [switch]$Phase2,

    [switch]$Clean

)



$Python = "C:\Users\Lukas\miniconda3\envs\mgr\python.exe"

$env:PYTHONIOENCODING = "utf-8"

Set-Location $PSScriptRoot



if (-not (Test-Path $Python)) {

    Write-Host "ERROR: Python not found at $Python" -ForegroundColor Red

    Write-Host "Activate conda env 'mgr' or update the path in run_windows.ps1"

    exit 1

}



if ($Clean) {

    if ($Phase2) {

        Write-Host "Removing Phase 2 checkpoints..." -ForegroundColor Yellow

        if (Test-Path "checkpoints/baseline_monai") { Remove-Item -Recurse -Force "checkpoints/baseline_monai" }
        if (Test-Path "checkpoints/hybrid_monai") { Remove-Item -Recurse -Force "checkpoints/hybrid_monai" }
        if (Test-Path "checkpoints/selective_net_monai") { Remove-Item -Recurse -Force "checkpoints/selective_net_monai" }
        if (Test-Path "checkpoints/evidential_monai") { Remove-Item -Recurse -Force "checkpoints/evidential_monai" }

    } elseif (Test-Path "checkpoints") {

        Write-Host "Removing checkpoints..." -ForegroundColor Yellow

        Remove-Item -Recurse -Force "checkpoints"

    }

}



if ($Verify -or (-not $PrepareData -and -not $TrainAll -and -not $Evaluate)) {

    Write-Host "`n[1] Verifying environment..." -ForegroundColor Green

    & $Python scripts/verify_env.py

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    if ($Phase2) {

        Write-Host "`n[1b] Verifying MONAI backbone..." -ForegroundColor Green

        & $Python scripts/verify_monai_backbone.py

        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    }

}



if ($PrepareData) {

    Write-Host "`n[2] Preparing ADNI dataset..." -ForegroundColor Green

    & $Python scripts/prepare_adni_dataset.py

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

}



if ($TrainAll) {

    $phaseLabel = if ($Phase2) { "Phase 2 (MONAI)" } else { "Phase 1" }

    Write-Host "`n[3] Training all models ($phaseLabel)..." -ForegroundColor Green

    $trainArgs = @("scripts/check_and_train_all.py")

    if ($Phase2) { $trainArgs += "--phase2" }

    if ($Clean) { $trainArgs += "--force" }

    & $Python @trainArgs

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n[3b] Training summary..." -ForegroundColor Green

    $summaryArgs = @("scripts/training_summary.py")

    if ($Phase2) { $summaryArgs += "--phase2" }

    & $Python @summaryArgs

}



if ($Evaluate) {

    Write-Host "`n[4] Evaluating on test set..." -ForegroundColor Green

    $evalArgs = @("scripts/evaluate_all.py")

    if ($Phase2) { $evalArgs += "--include-phase2" }

    & $Python @evalArgs

    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

}



Write-Host "`nDone." -ForegroundColor Green

