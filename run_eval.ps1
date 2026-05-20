# AI-GENERATED - SAM3 Evaluation Launcher
# Sets HuggingFace token and runs evaluation

param(
    [string]$Token = "",
    [int]$Samples = 1
)

# Check if token is provided
if ([string]::IsNullOrEmpty($Token)) {
    Write-Host "❌ Error: HuggingFace token required" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage: .\run_eval.ps1 -Token 'hf_your_token_here' [-Samples 10]"
    Write-Host ""
    Write-Host "To get a token:" -ForegroundColor Yellow
    Write-Host "  1. Go to: https://huggingface.co/settings/tokens"
    Write-Host "  2. Accept license: https://huggingface.co/facebook/sam3"
    Write-Host "  3. Create token (select 'repo' type)"
    Write-Host ""
    exit 1
}

Write-Host "🔐 Setting HuggingFace authentication..." -ForegroundColor Green
Write-Host ""

# Set environment variable for this process and children
$env:HF_TOKEN = $Token

# Also store for future sessions
Write-Host "💾 Saving token for future sessions..." -ForegroundColor Cyan
try {
    [System.Environment]::SetEnvironmentVariable('HF_TOKEN', $Token, 'User')
    Write-Host "✓ Token saved to user environment variables" -ForegroundColor Green
} catch {
    Write-Host "⚠ Could not save to permanent storage: $_" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Starting evaluation..." -ForegroundColor Green
Write-Host "   Samples: $Samples" -ForegroundColor Cyan
Write-Host ""

# Change to project directory
cd "C:\Users\Habib\Desktop\CV\project\SPARE"

# Run evaluation
uv run python scripts/evaluate_robustness.py $Samples

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Evaluation completed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Results saved to: experiments/robustness_eval_2026-05-07/" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Evaluation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
