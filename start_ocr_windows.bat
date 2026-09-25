@echo off
REM Best local Windows profile from bench_configs.py (accuracy = full PP-OCRv4, 0 crashes)
cd /d "%~dp0"

set OCR_LOW_MEMORY=false
set OCR_FAST_MODE=false
set OCR_ENABLE_MKLDNN=false
set OCR_VERSION=PP-OCRv4
set OCR_DPI=150
set OCR_MAX_SIDE=2400
set OCR_DET_LIMIT=960
set OCR_CPU_THREADS=1
set OCR_REC_BATCH_NUM=1
set OCR_USE_ANGLE_CLS=false
set OCR_MAX_PAGES=1

"C:\Users\Admin\AppData\Local\Programs\Python\Python310\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8080
