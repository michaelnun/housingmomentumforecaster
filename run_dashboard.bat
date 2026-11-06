@echo off
REM Double-click to open the dashboard in your browser.
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
)
python -c "import streamlit, plotly" 2>nul || python -m pip install --quiet streamlit plotly
if not exist results\index_history.csv (
  echo No real results yet, building the prototype on simulated data...
  python make_demo_data.py
  python step2_build.py --raw data/raw_demo
  python step3_model.py
  python step4_index.py
)
python -m streamlit run dashboard.py
pause
