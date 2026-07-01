@echo off
REM ===========================================================================
REM Khoi dong GIAO DIEN WEB chinh (app.py):
REM   OCR -> xoa phu de -> dich -> cay phu de -> long tieng.
REM Chay trong moi truong conda "ste" (xem index.md muc 3).
REM Long tieng se goi sang service OmniVoice (chay run_omnivoice.bat tr-uoc).
REM ===========================================================================
chcp 65001 >nul
set PYTHONUTF8=1

REM Doi cong web neu can (mac dinh 7860)
if "%GRADIO_PORT%"=="" set GRADIO_PORT=7860

call conda activate ste
if errorlevel 1 (
    echo [LOI] Khong activate duoc moi truong conda "ste".
    echo       Tao moi truong theo index.md muc 3 truoc nhe.
    pause
    exit /b 1
)

python app.py
pause
