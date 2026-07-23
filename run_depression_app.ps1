Set-Location "C:\Users\sreen\Downloads\project\Depression_detection"

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found."
    exit 1
}

Start-Process chrome "http://127.0.0.1:8000/"
uvicorn webapp.main:app --reload --port 8000
