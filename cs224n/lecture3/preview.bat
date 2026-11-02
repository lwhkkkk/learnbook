@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动 Stanford CS224N Lecture 3 电子书预览...
echo 访问地址: http://localhost:8081/book.html
echo (按 Ctrl+C 可停止退出)
start http://localhost:8081/book.html
python -m http.server 8081
if errorlevel 1 (
    d:\project\videobook\.venv\Scripts\python.exe -m http.server 8081
)
pause
