@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto :dependencias
where py >nul 2>nul
if errorlevel 1 goto :via_python
py -3 -m venv ".venv"
if not errorlevel 1 goto :dependencias
:via_python
python -m venv ".venv"
if not errorlevel 1 goto :dependencias
echo Instale o Python 3.11 ou mais recente de 64 bits, incluindo Tcl/Tk e pip.
goto :falha
:dependencias
set "PYTHON=.venv\Scripts\python.exe"
"%PYTHON%" -c "import sys; sys.exit(sys.version_info < (3, 11))"
if errorlevel 1 (
    echo Esta pasta requer Python 3.11 ou mais recente. Consulte o README.md.
    goto :falha
)
"%PYTHON%" -c "import tkinter, numpy, scipy.ndimage, cv2, matplotlib, PIL.ImageTk" >nul 2>nul
if not errorlevel 1 goto :executar
echo Preparando os pacotes. Na primeira vez, mantenha a internet conectada.
"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 goto :falha
:executar
"%PYTHON%" main.py
if errorlevel 1 goto :falha
exit /b 0
:falha
echo Nao foi possivel iniciar o TrapMotion. A mensagem acima explica o motivo.
pause
exit /b 1
