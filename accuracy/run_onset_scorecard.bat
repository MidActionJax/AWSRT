@echo off
REM Hourly live scorecard for the onset-risk forecast (scheduled task
REM "Justified_Adaptive_fluX_onset_scorecard"). Logs one row per lead time;
REM duplicate issue times are skipped. Output appended to onset_scorecard.log.
cd /d "C:\Users\jaxon\OneDrive\Desktop\Github\AWSRT\accuracy"
echo ===== %DATE% %TIME% >> onset_scorecard.log
"C:\Users\jaxon\AppData\Local\Programs\Python\Python310\python.exe" onset_scorecard.py --hours 6 12 24 >> onset_scorecard.log 2>&1
