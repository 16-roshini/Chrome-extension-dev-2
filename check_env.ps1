# Environment check script
$venvPython = 'C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe'
$venvPip = 'C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\pip.exe'

Write-Host "=== Python Version ===" -ForegroundColor Cyan
& $venvPython --version

Write-Host "`n=== Installed Packages ===" -ForegroundColor Cyan
& $venvPip list

Write-Host "`n=== Tesseract Installation ===" -ForegroundColor Cyan
$tessPath = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
if (Test-Path $tessPath) {
    Write-Host "Tesseract FOUND at: $tessPath" -ForegroundColor Green
    & $tessPath --version 2>&1
} else {
    Write-Host "Tesseract NOT FOUND at expected path" -ForegroundColor Red
}

Write-Host "`n=== spaCy model check ===" -ForegroundColor Cyan
& $venvPython -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('en_core_web_sm: FOUND')" 2>&1

Write-Host "`n=== EasyOCR check ===" -ForegroundColor Cyan
& $venvPython -c "import easyocr; print('easyocr: FOUND')" 2>&1

Write-Host "`n=== pytesseract check ===" -ForegroundColor Cyan
& $venvPython -c "import pytesseract; print('pytesseract: FOUND')" 2>&1

Write-Host "`n=== FastAPI check ===" -ForegroundColor Cyan
& $venvPython -c "import fastapi; print('fastapi version:', fastapi.__version__)" 2>&1

Write-Host "`n=== Done ===" -ForegroundColor Cyan
