# Legacy entry point — delegates to run_windows.ps1 (Windows, no WSL)
param(
    [switch]$Clean
)

if ($Clean) {
    & "$PSScriptRoot\run_windows.ps1" -Clean -PrepareData -TrainAll
} else {
    & "$PSScriptRoot\run_windows.ps1" -PrepareData -TrainAll
}
