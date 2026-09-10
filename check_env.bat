@echo off
echo === Python Version ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" --version

echo.
echo === Pip List ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\pip.exe" list

echo.
echo === Tesseract ===
if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
    echo Tesseract FOUND
    "C:\Program Files\Tesseract-OCR\tesseract.exe" --version
) else (
    echo Tesseract NOT FOUND
)

echo.
echo === spaCy model ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('en_core_web_sm: FOUND')"

echo.
echo === EasyOCR ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import easyocr; print('easyocr: FOUND')"

echo.
echo === pytesseract ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import pytesseract; print('pytesseract: FOUND')"

echo.
echo === FastAPI ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import fastapi; print('fastapi:', fastapi.__version__)"

echo.
echo === pytest ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import pytest; print('pytest:', pytest.__version__)"

echo.
echo === pydantic ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import pydantic; print('pydantic:', pydantic.__version__)"

echo.
echo === phonenumbers ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import phonenumbers; print('phonenumbers: FOUND')"

echo.
echo === stdnum ===
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "import stdnum; print('python-stdnum: FOUND')"

echo.
echo === Done ===
