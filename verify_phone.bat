@echo off
"C:\Users\chila\OneDrive\Desktop\ocr and pii detector\venv\Scripts\python.exe" -c "
from server.ocr_pii.detectors.patterns import PHONE

tests = [
    ('+91 98765 43210', True),
    ('9876543210', True),
    ('+91-9876543210', True),
    ('91 9876543210', True),
    ('+91 9876543210', True),
    ('98765 43210', True),
    ('1234567890', False),
    ('123456', False),
]

all_ok = True
for text, should_match in tests:
    m = PHONE.search(text)
    matched = m is not None
    status = 'OK' if matched == should_match else 'FAIL'
    if status == 'FAIL':
        all_ok = False
    print(f'  [{status}] {text!r:25s} expected={should_match} got={matched}')

print()
print('All OK!' if all_ok else 'Some tests FAILED')
"
