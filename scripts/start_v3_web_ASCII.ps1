param([int]$Port=8000,[switch]$Offline)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$py=(Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CMC_DATA_MODE='DEMO'
Remove-Item Env:CMC_ALLOW_BASELINE -ErrorAction SilentlyContinue
if ($Offline) { $env:CMC_RAG_CLOUD='0'; $env:CMC_CHAT_AGENT='0' } else {
    $env:CMC_RAG_CLOUD='1'; $env:CMC_CHAT_AGENT='1'
    & $py -B -c "import os; from dotenv import load_dotenv; load_dotenv('.env'); assert os.getenv('GEMINI_API_KEY'), 'Gemini key is missing; configure private .env first.'"
    if ($LASTEXITCODE -ne 0) {throw 'Gemini key missing. No server started.'}
    Write-Host 'V3 DEMO: typed questions may be sent to Gemini. Never type real personal data or secrets.'
}
Write-Host "Open http://127.0.0.1:$Port in your browser. Ctrl+C stops the server."
& $py -B -X utf8 -m uvicorn app:app --host 127.0.0.1 --port $Port --no-proxy-headers --no-use-colors
if ($LASTEXITCODE -ne 0) {throw 'Uvicorn exited with an error.'}
