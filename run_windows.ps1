# run_windows.ps1 — uruchamianie projektu na Windows bez WSL

param(
    [switch]$Verify,
    [switch]$PrepareData,
    [switch]$TrainAll,
    [switch]$Evaluate,
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
    Write-Host "Removing checkpoints..." -ForegroundColor Yellow
    foreach ($dir in @("baseline", "hybrid", "selective_net", "evidential")) {
        $path = "checkpoints/$dir"
        if (Test-Path $path) { Remove-Item -Recurse -Force $path }
    }
}

if ($Verify -or (-not $PrepareData -and -not $TrainAll -and -not $Evaluate)) {
    Write-Host "`n[1] Verifying environment..." -ForegroundColor Green
    & $Python scripts/verify_env.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n[1b] Verifying MONAI backbone..." -ForegroundColor Green
    & $Python scripts/verify_monai_backbone.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($PrepareData) {
    Write-Host "`n[2] Preparing ADNI dataset..." -ForegroundColor Green
    & $Python scripts/prepare_dataset.py --dataset_root "Data baseline" --output data_metadata_adni.csv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if ($TrainAll) {
    Write-Host "`n[3] Training all models (MONAI)..." -ForegroundColor Green
    $trainArgs = @("scripts/check_and_train_all.py")
    if ($Clean) { $trainArgs += "--force" }
    & $Python @trainArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n[3b] Training summary..." -ForegroundColor Green
    & $Python scripts/training_summary.py
}

if ($Evaluate) {
    Write-Host "`n[4] Evaluating on test set..." -ForegroundColor Green
    & $Python scripts/evaluate_all.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`nDone." -ForegroundColor Green
