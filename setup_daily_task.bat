@echo off
REM =====================================================================
REM  ViralOps Engine - Windows Task Scheduler Setup
REM  Creates a daily task to auto-publish 1 TikTok post at 7:00 AM
REM =====================================================================

set TASK_NAME=ViralOps-DailyPost
set PROJ_DIR=%~dp0
set PYTHON=python
set SCRIPT=%PROJ_DIR%daily_scheduler.py

echo.
echo  ViralOps - Task Scheduler Setup
echo  ================================
echo  Task:   %TASK_NAME%
echo  Script: %SCRIPT%
echo  Time:   07:00 AM daily
echo.

REM Delete existing task (if any)
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

REM Create new daily task at 7:00 AM
schtasks /Create ^
    /TN "%TASK_NAME%" ^
    /TR "\"%PYTHON%\" \"%SCRIPT%\"" ^
    /SC DAILY ^
    /ST 07:00 ^
    /RL HIGHEST ^
    /F

if %ERRORLEVEL% equ 0 (
    echo.
    echo  [OK] Task '%TASK_NAME%' created successfully!
    echo  Runs daily at 07:00 AM.
    echo.
    echo  To check:  schtasks /Query /TN "%TASK_NAME%" /V
    echo  To delete:  schtasks /Delete /TN "%TASK_NAME%" /F
    echo  To run now: schtasks /Run /TN "%TASK_NAME%"
) else (
    echo.
    echo  [ERROR] Failed to create task. Try running as Administrator.
)
echo.
pause
