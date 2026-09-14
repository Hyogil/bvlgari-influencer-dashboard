@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"

echo.
echo ==========================================
echo  BVLGARI Influencer Dashboard Launcher
echo ==========================================
echo.

set "PY_CMD="

where py >nul 2>&1
if %errorlevel%==0 (
    set "PY_CMD=py"
) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python이 설치되어 있지 않거나 PATH에 등록되어 있지 않습니다.
    echo.
    echo 1. https://www.python.org/downloads/windows/ 에서 Python 3.11 또는 3.12 설치
    echo 2. 설치 화면에서 반드시 "Add python.exe to PATH" 체크
    echo 3. 설치 완료 후 이 창을 닫고 run_windows_fixed.bat를 다시 실행
    echo.
    echo 참고: Microsoft Store의 Python App Execution Alias 때문에
    echo       가짜 python 명령이 잡히는 경우가 있습니다.
    echo       Settings ^> Apps ^> Advanced app settings ^> App execution aliases 에서
    echo       python.exe / python3.exe 토글을 끄면 됩니다.
    echo.
    pause
    exit /b 1
)

echo [OK] Python command: %PY_CMD%
%PY_CMD% --version

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo [1/4] 가상환경 생성 중...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto :error
)

echo.
echo [2/4] pip 업데이트 중...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [3/4] 필요한 패키지 설치 중...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo [4/4] 웹 서버 시작...
echo 브라우저 주소: http://127.0.0.1:8000
echo.

start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app:app --reload

goto :eof

:error
echo.
echo [ERROR] 실행 중 오류가 발생했습니다.
echo 위 메시지를 복사해서 보내주시면 다음 단계로 바로 확인할 수 있습니다.
echo.
pause
exit /b 1
