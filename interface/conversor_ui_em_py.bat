@echo off
title .ui to .py file converter!
echo Python file generator from .ui files!
pause
echo .
echo .
echo .
echo .ui file name (with .ui included)
set /p UiName=Enter .UI file Name: 
echo .
echo .
echo .
echo .py file name (with .py included)
set /p PyName=Enter .PY file Name: 
echo .
echo .
echo .
echo Starting file conversion, please wait.

call pyuic5 -x -o "%PyName%" "%UiName%"

echo The conversion was completed with Success.
pause