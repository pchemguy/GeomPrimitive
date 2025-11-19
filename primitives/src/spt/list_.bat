@echo off
setlocal enableextensions enabledelayedexpansion

rem ============================================================
rem  COMBINE PYTHON MODULES FOR LLM OVERVIEW
rem  Output: combined_for_llm.txt
rem ============================================================

set OUTPUT=combined_for_llm.txt
if exist "%OUTPUT%" del "%OUTPUT%"

rem ---- List of modules to combine ----------------------------
set FILES= ^
spt_geometry.py ^
spt_lighting.py ^
spt_noise.py ^
spt_pipeline.py ^
spt_texture.py



rem ------------------------------------------------------------
echo Combining modules...
echo =========================================================== >> "%OUTPUT%"

for %%F in (%FILES%) do (
    if exist "%%F" (
        echo ===== START %%F ===== >> "%OUTPUT%"
        echo ```python >> "%OUTPUT%"
        type "%%F" >> "%OUTPUT%"
        echo ``` >> "%OUTPUT%"
        echo. >> "%OUTPUT%"
        echo ===== END %%F ===== >> "%OUTPUT%"
        echo. >> "%OUTPUT%"
    ) else (
        echo [WARN] File not found: %%F
    )
)

echo Done.
echo Output written to: %OUTPUT%

endlocal
