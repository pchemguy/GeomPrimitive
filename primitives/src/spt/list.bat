@echo off
setlocal enableextensions enabledelayedexpansion

rem ============================================================
rem  COMBINE PYTHON MODULES FOR LLM OVERVIEW
rem  Output: combined_for_llm.txt
rem ============================================================

set OUTPUT=combined_for_llm.txt
if exist "%OUTPUT%" del "%OUTPUT%"

rem ---- List of modules to combine ----------------------------
set FILES=fqc.py ^
mpl_artist_preview.py ^
mpl_grid_gen.py ^
mpl_grid_gen_demo.py ^
mpl_grid_gen_effects.py ^
mpl_grid_gen_pipeline.py ^
mpl_grid_utils~.py ^
mpl_path_utils.py ^
mpl_renderer.py ^
mpl_utils.py ^
spt.py ^
spt_color.py ^
spt_config.py ^
spt_correction_engine.py ^
spt_correction_engine_random.py ^
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
