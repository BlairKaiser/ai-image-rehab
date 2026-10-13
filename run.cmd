@echo off
rem Start the page and open it in the browser. Close this window to stop it.
cd /d "%~dp0"
python -I app\ripple_studio.py --open %*
pause
