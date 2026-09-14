@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo BVLGARI Influencer Dashboard
echo ==========================================
echo.

set "BASEPY="

REM 1) Prefer the known per-user Python 3.12 install location.
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "BASEPY=%LocalAppData%\Programs\Python\Python312\python.exe"
)

REM 2) Try Python 3.13 if 3.12 was not found.
if not defined BASEPY if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "BASEPY=%LocalAppData%\Programs\Python\Python313\python.exe"
)

REM 3) Try Python 3.11 if needed.
if not defined BASEPY if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "BASEPY=%LocalAppData%\Programs\Python\Python311\python.exe"
)

REM 4) Fall back to py launcher.
if not defined BASEPY (
    where py >nul 2>&1
    if not errorlevel 1 set "BASEPY=py"
)

if not defined BASEPY (
    echo ERROR: Python was not found.
    echo Expected path example:
    echo   %LocalAppData%\Programs\Python\Python312\python.exe
    echo.
    pause
    exit /b 1
)

echo Using Python:
if /i "%BASEPY%"=="py" (
    py -3.12 --version >nul 2>&1
    if not errorlevel 1 (
        py -3.12 --version
        set "CREATE_VENV=py -3.12"
    ) else (
        py --version
        set "CREATE_VENV=py"
    )
) else (
    "%BASEPY%" --version
    set "CREATE_VENV="%BASEPY%""
)

echo.

REM Recreate the virtual environment only if it does not exist.
if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating virtual environment...
    if /i "%BASEPY%"=="py" (
        %CREATE_VENV% -m venv .venv
    ) else (
        "%BASEPY%" -m venv .venv
    )
    if errorlevel 1 goto :fail
) else (
    echo [1/4] Virtual environment already exists.
)

echo [2/4] Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail

echo [3/4] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [4/4] Starting web server...
echo.
echo Open this URL if the browser does not open automatically:
echo   http://127.0.0.1:8000
echo.

start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app:app --reload

exit /b 0

:fail
echo.
echo ERROR: Startup failed.
echo Please send the error lines shown above.
echo.
pause
exit /b 1
