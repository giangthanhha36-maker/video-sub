@echo off
REM ===========================================================================
REM Khoi dong SERVICE OmniVoice (long tieng) - chay o moi truong RIENG.
REM Service nay nap model 1 lan va giu trong VRAM -> nhieu video dung lai
REM khoi nap lai -> tiet kiem thoi gian/tien GPU.
REM
REM Lan dau tao moi truong (xem index.md muc "Long tieng OmniVoice"):
REM   conda create -n omnivoice python=3.12 -y
REM   conda activate omnivoice
REM   pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 torchvision==0.23.0+cu128 ^
REM       --extra-index-url https://download.pytorch.org/whl/cu128
REM   pip install -r requirements-omnivoice.txt
REM ===========================================================================
chcp 65001 >nul
set PYTHONUTF8=1

call conda activate omnivoice
if errorlevel 1 (
    echo [LOI] Khong activate duoc moi truong conda "omnivoice".
    echo       Tao moi truong theo index.md muc "Long tieng OmniVoice" truoc nhe.
    pause
    exit /b 1
)

REM Ghi chu: KHONG bat --no-asr de voice clone van chay duoc ngay ca khi
REM ban bo trong "Reference Text". Neu LUON nhap san reference text va muon
REM nhe VRAM + khoi dong nhanh hon, them " --no-asr" vao cuoi lenh duoi.
python audio.py --model k2-fsa/OmniVoice --device cuda --ip 0.0.0.0 --port 7861
pause
