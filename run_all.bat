@echo off
REM Double-click this file in File Explorer.
cd /d "%~dp0"
echo Working... the first run after adding metros takes 1.5-2 hours (mostly Google Trends). Later monthly runs take about an hour.
echo Everything is also saved to run_log.txt in this folder, so Claude can check progress.
echo Leave this window open.
echo.
call :main > run_log.txt 2>&1
type run_log.txt | more
echo.
pause
exit /b

:main
echo --- started %date% %time% ---
REM Anaconda's Python only works properly after "activate", so do that first.
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
)
where python
python --version || (echo PYTHON NOT FOUND & exit /b 1)
echo === Installing packages ===
python -m pip install --quiet pandas numpy scikit-learn scipy matplotlib requests pytrends streamlit plotly || (echo PIP FAILED & exit /b 1)
python -c "import pandas, requests, sklearn; print('imports ok')" || (echo IMPORT FAILED & exit /b 1)
echo === Downloading real data ===
python -u step1_collect.py || (echo STEP 1 FAILED - double-click again to resume & exit /b 1)
echo === Building the table and testing the models ===
python step2_build.py || (echo STEP 2 FAILED & exit /b 1)
python step3_model.py || (echo STEP 3 FAILED & exit /b 1)
python step4_index.py || (echo STEP 4 FAILED & exit /b 1)
python step5_activity.py || echo STEP 5 (activity test) FAILED - dashboard still works
python step6_findings.py || echo STEP 6 (findings) FAILED - dashboard still works
python step7_signal_tests.py || echo STEP 7 (signal tests) FAILED - dashboard still works
python step8_robustness.py || echo STEP 8 (robustness) FAILED - dashboard still works
python step9_stability.py || echo STEP 9 (stability) FAILED - dashboard still works
python step10_search.py || echo STEP 10 (search) FAILED - dashboard still works
python step11_news.py || echo STEP 11 (news) FAILED - dashboard still works
echo DONE %date% %time%. Open the dashboard with run_dashboard.bat
exit /b 0
