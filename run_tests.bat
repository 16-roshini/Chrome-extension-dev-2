@echo off
cd /d "C:\Users\chila\OneDrive\Desktop\ocr and pii detector"
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -m pytest server/ocr_pii/tests/test_regex_detector.py server/ocr_pii/tests/test_ner_detector.py server/ocr_pii/tests/test_context_detector.py server/ocr_pii/tests/test_aggregator.py -v --tb=short 2>&1
