@echo off
chcp 65001 >nul
cd /d %~dp0
python amp_benchmark_menu.py
pause
