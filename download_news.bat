@echo off
REM Double-click to download the news data (GDELT) on this computer. Takes about 15-30 minutes.
REM Afterwards: in GitHub Desktop, commit and push, then run the monthly update on GitHub.
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
)
python -c "import pandas, requests" 2>nul || python -m pip install --quiet pandas requests
python step1_collect.py --news-only
pause
