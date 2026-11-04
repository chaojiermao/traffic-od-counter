@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 车型识别模型服务 (localhost:8502)
echo ============================================================
echo   车型识别模型服务
echo   网页里的「模型计数」会连 http://127.0.0.1:8502
echo   这个窗口不要关，关了网页就连不上模型
echo ============================================================
echo.
if not exist "venv\Scripts\python.exe" (
    echo [ERROR] 没找到 venv，请先执行: pip install -r requirements.txt
    pause
    exit /b 1
)
venv\Scripts\python.exe model_server.py 8502
echo.
echo 服务已停止
pause
