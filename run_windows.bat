@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo BVLGARI Influencer Dashboard
echo ==========================================
echo.

set "BASEPY="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "BASEPY=%LocalAppData%\Programs\Python\Python312\python.exe"
if not defined BASEPY if exist "%LocalAppData%\Programs\Python\Python313\python.exe" set "BASEPY=%LocalAppData%\Programs\Python\Python313\python.exe"
if not defined BASEPY if exist "%LocalAppData%\Programs\Python\Python311\python.exe" set "BASEPY=%LocalAppData%\Programs\Python\Python311\python.exe"
if not defined BASEPY (
  where py >nul 2>&1
  if not errorlevel 1 set "BASEPY=py"
)
if not defined BASEPY (
  echo ERROR: Python was not found.
  echo Expected path example:
  echo   %LocalAppData%\Programs\Python\Python312\python.exe
  pause
  exit /b 1
)

echo Using Python:
if /i "%BASEPY%"=="py" (py --version) else ("%BASEPY%" --version)
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating virtual environment...
  if /i "%BASEPY%"=="py" (py -m venv .venv) else ("%BASEPY%" -m venv .venv)
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
pause
exit /b 1
