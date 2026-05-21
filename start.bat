@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo.
echo ================================
echo   AI Assistant 启动中...
echo ================================
echo.
echo   浏览器将自动打开 http://localhost:8000
echo   关闭此窗口即可停止服务
echo.

start "" http://localhost:8000

python api_server.py

pause
