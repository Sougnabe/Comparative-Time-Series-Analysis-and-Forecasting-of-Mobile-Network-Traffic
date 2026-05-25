# One-click runner for the main pipeline (sample run)
# Edit or run in PowerShell after activating the venv.

Write-Host "Running sample pipeline..."
$python = ".\.venv\Scripts\python.exe"

# 1. Run EDA
& $python scripts/run_eda.py

# 2. Train models (may be slow on full dataset)
& $python scripts/train_models.py

# 3. Generate diagnostics
& $python scripts/plot_model_diagnostics.py

# 4. (omitted)

Write-Host "Pipeline completed. Check artifacts/ for outputs."