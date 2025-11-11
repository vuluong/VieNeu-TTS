@echo off
setlocal
cd /d %~dp0\..
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip wheel
REM Pick one CUDA line:
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
REM pip install flash-attn==2.7.3 --no-build-isolation
echo [OK] Environment ready. Use: .venv\Scripts\activate
