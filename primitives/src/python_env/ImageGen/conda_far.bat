@echo off

:: ============================================================================
::  Purpose:
::    Orchestrates initialization and activation of a minimal Conda-based
::    development environment for Windows-based FFCV/fastxtend builds.
::    Ensures correct setup of MS Build Tools, Conda/Micromamba environment,
::    and native library dependencies (pthreads, OpenCV, libjpeg-turbo).
::
::  Description:
::    This script serves as the main entry point for environment activation
::    and dependency management. It guarantees that:
::      - cmd.exe Delayed Expansion is enabled
::      - The shell is free from preactivated Python/Conda contexts
::      - MS Build Tools are available and activated
::      - Required libraries are initialized via their respective scripts
::      - Proper INCLUDE, LIB, LINK, and PATH variables are configured
::      - Environment is ready for subsequent FFCV installation or builds
::
::  Invocation Modes:
::      /batch        - Activates environment variables only; does not launch
::                      FAR Manager or start a new interactive shell.
::
::      /preactivate  - Performs environment pre-initialization
::
::      (no argument) - Activates full environment and launches FAR Manager
::                      (if detected) or opens a regular cmd.exe session.
::
::  Behavioral Summary:
::      1. Verifies cmd.exe configuration and base environment cleanliness.
::      2. Activates MS Build Tools environment or notifies user of failure.
::      3. Ensures Conda (or Micromamba) environment readiness.
::      4. Sequentially activates pthreads, OpenCV, and libjpeg-turbo.
::      5. Updates INCLUDE, LIB, and LINK paths to integrate Conda libraries.
::      6. Exposes DISTUTILS_USE_SDK=1 to enable setuptools to reuse the
::         existing MSVC environment instead of launching new compiler shells.
::      7. Starts FAR Manager if available, or leaves the user in a prepared
::         cmd.exe session.
::
::  Exit Codes:
::      0   - Success
::      1+  - Failure during activation (refer to last console output)
::
::  Notes:
::      - Requires Windows 10+ with ANSI color output support.
::      - Requires curl.exe and tar.exe in PATH (included by default in Win10+).
::      - Colorized output can be disabled by defining NOCOLOR=1.
:: ============================================================================

:: --- Parse arguments and preserve top-level context ---

call :PARSE_ARGS %*

:: --- Escape sequence templates for color coded console output ---

call :COLOR_SCHEME

echo:
echo ==========================================================================
echo %INFO% Setting up environment
echo %INFO% 
echo %WARN% CLI: "%~f0" %*
echo ==========================================================================
echo:

set "EXIT_STATUS=1"

:: --- Default Conda Prefix ---

if defined _CONDA_PREFIX (
  set "__CONDA_PREFIX=%_CONDA_PREFIX%"
) else (
  set "__CONDA_PREFIX=%~dp0Anaconda"
)
set "CONDA_BAT=%__CONDA_PREFIX%\condabin\conda.bat"
set "MAMBA_BAT=%__CONDA_PREFIX%\condabin\mamba.bat"

:: --- Make sure cmd.exe delayed expansion is enabled by default ---

call :CHECK_DELAYED_EXPANSION
if not "%ERRORLEVEL%"=="0" if defined _ARGS (

  rem -- Delayed Expansion is disabled, running in non-interactive mode ---
  
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Delayed Expansion is disabled while running in non-interactive mode. Aborting...
  goto :CLEANUP
) else (

  rem --- Delayed Expansion is disabled, running in interactive mode {no arguments supplied} ---
  
  setlocal EnableDelayedExpansion EnableExtensions
)

:: --- Determine cache directory ---

if not defined _CACHE (
  call :CACHE_DIR
  set "EXIT_STATUS=!ERRORLEVEL!"
) else (
  set "EXIT_STATUS=0"
)
if not defined _CACHE (
  echo %ERROR% Failed to set CACHE directory. Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
echo:

:: --- Base / Root environment guard ---

call :NO_ROOT_ENVIRONMENT
if not "%ERRORLEVEL%"=="0" (
  echo %ERROR% Aborting due to pre-existing Python/Conda environment.
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  goto :CLEANUP
)

:: --- Python.exe and conda.bat must exist in Conda environment ---

if not exist "%__CONDA_PREFIX%\python.exe" (
  echo %ERROR% Python not found: "%__CONDA_PREFIX%\python.exe". Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
if not exist "%CONDA_BAT%" (
  echo %ERROR% Conda activation script not found: "%CONDA_BAT%". Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
if not exist "%MAMBA_BAT%" (
  echo %ERROR% Conda activation script not found: "%MAMBA_BAT%". Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)

:: --- Activate Conda environment ---

echo:
echo %WARN% Activating Conda PREFIX "%__CONDA_PREFIX%".
call "%CONDA_BAT%" activate
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Conda environment activation failed. Aborting...
  goto :CLEANUP
)
if not exist "!CONDA_PREFIX!\python.exe" (
  echo %ERROR% Conda environment activation failed - Python "!CONDA_PREFIX!\python.exe" not found. Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
echo %OKOK% Conda activation succeeded.

:: --- Use "/batch" to activate shell environment without starting FAR MANAGER ---

set "FINAL_EXIT_CODE=0"
if defined _ARG_BATCH goto :CLEANUP
if defined _ARG_PREACTIVATE goto :CLEANUP

:: --- Start FAR MANAGER ---

for %%A in ("far.bat" "far.exe") do (
  where /Q %%~A >nul 2>nul && (
    set "_FARMANAGER=%%~A"
    goto :START_FARMANAGER
  )
)

:START_FARMANAGER

if not defined _FARMANAGER set "_FARMANAGER=cd"
cmd /E:ON /V:ON /K "%_FARMANAGER% ""%__CONDA_PREFIX%"""
set "_FARMANAGER="

goto :CLEANUP
:: ============================================================================
:: ============================================================================
:: ============================================================================


:: ============================================================================ COLOR_SCHEME BEGIN
:: ============================================================================
:COLOR_SCHEME
:: ---------------------------------------------------------------------
:: Color Scheme (with NOCOLOR fallback)
:: ---------------------------------------------------------------------

if defined _ARG_NOCOLOR set "NOCOLOR=1"
if defined NOCOLOR (
  set  "INFO= [INFO]  "
  set  "OKOK= [-OK-]  "
  set  "WARN= [WARN]  "
  set "ERROR= [ERROR] "
) else (
  set  "INFO=[100;92m [INFO]  [0m"
  set  "OKOK=[103;94m [-OK-]  [0m"
  set  "WARN=[106;35m [WARN]  [0m"
  set "ERROR=[105;34m [ERROR] [0m"
)

exit /b 0
:: ============================================================================ 
:: ============================================================================ COLOR_SCHEME END


:: ============================================================================ PARSE_ARGS BEGIN
:: ============================================================================
:: --- Parse arguments ---
:: Because "shift" destroys %0, original context is preserved by destroying this.

:: --- Parsing arguments ---

:PARSE_ARGS

if not "%~1"=="" (
  set "_ARGS=TRUE"
) else (
  set "_ARGS="
  goto :PARSE_ARGS_DONE
)

set "_ARG_BATCH="
set "_ARG_PREACTIVATE="
set "_MODE="

:PARSE_NEXT_ARG

if /I "%~1"=="" goto :PARSE_ARGS_DONE
if /I "%~1"=="/batch"       set "_ARG_BATCH=1"
if /I "%~1"=="/preactivate" set "_ARG_PREACTIVATE=1"
if /I "%~1"=="/nocolor"     set "_ARG_NOCOLOR=1"
shift
goto :PARSE_NEXT_ARG

:PARSE_ARGS_DONE

exit /b 0
:: ============================================================================ 
:: ============================================================================ PARSE_ARGS END


:: ============================================================================ CLEANUP BEGIN
:: ============================================================================
:: --- Clean up; prefer as the primary script exit point ---
:: To exit script, set FINAL_EXIT_CODE and goto CLEANUP

:CLEANUP

set "_ARGS="
set "_ARG_BATCH="
set "_ARG_PREACTIVATE="
set "_MODE="
set  "INFO="
set  "OKOK="
set  "WARN="
set "ERROR="
set "EXIT_STATUS="
set "__CONDA_PREFIX="

:: --- Ensure a valid exit code is always returned ---

if not defined FINAL_EXIT_CODE set "FINAL_EXIT_CODE=1"
exit /b %FINAL_EXIT_CODE%
:: ============================================================================ 
:: ============================================================================ CLEANUP END


:: ============================================================================ CACHE_DIR BEGIN
:: ============================================================================
:CACHE_DIR
:: --------------------------------------------------------
:: Determine cache directory
:: --------------------------------------------------------
if exist "%_CACHE%" (
  goto :CACHE_DIR_SET
) else (
  set "_CACHE=%TEMP%"
)

if exist "%~d0\CACHE" (
  set "_CACHE=%~d0\CACHE"
  goto :CACHE_DIR_SET
)

if exist "%~dp0CACHE" (
  set "_CACHE=%~dp0CACHE"
  goto :CACHE_DIR_SET
)

if exist "%USERPROFILE%\Downloads" (
  if exist "%USERPROFILE%\Downloads\CACHE" (
    set "_CACHE=%USERPROFILE%\Downloads\CACHE"
  ) else (
    set "_CACHE=%USERPROFILE%\Downloads"
  )
  goto :CACHE_DIR_SET
)

:CACHE_DIR_SET
:: --------------------------------------------------------
:: Verify file system access
:: --------------------------------------------------------
set "_DUMMY=%_CACHE%\$$$_DELETEME_ACCESS_CHECK_$$$"
if exist "%_DUMMY%" rmdir /Q /S "%_DUMMY%"
set "EXIT_STATUS=%ERRORLEVEL%"
if exist "%_DUMMY%" set "EXIT_STATUS=1"
if not "!EXIT_STATUS!"=="0" (
  echo %ERROR% Failed to delete test directory "%_DUMMY%".
  echo %ERROR% Expected a full-access at this location "%_CACHE%".
  echo %ERROR% Aborting...
  set "_CACHE="
  exit /b !EXIT_STATUS!
)

md "%_DUMMY%"
set "EXIT_STATUS=%ERRORLEVEL%"
if not exist "%_DUMMY%" set "EXIT_STATUS=1"
if not "!EXIT_STATUS!"=="0" (
  echo %ERROR% Failed to create test directory "%_DUMMY%".
  echo %ERROR% Expected a full-access at this location "%_CACHE%".
  echo %ERROR% Aborting...
  set "_CACHE="
  exit /b !EXIT_STATUS!
)

:: --------------------------------------------------------
:: Point CONDA_PKGS_DIRS and PIP_CACHE_DIR to package cache directory
:: --------------------------------------------------------
set "_PKGS_DIR=%_CACHE%\Python\pkgs"

if not defined CONDA_PKGS_DIRS (
  set "CONDA_PKGS_DIRS=%_PKGS_DIR%"
) else (
  set "_PKGS_DIR=%CONDA_PKGS_DIRS%"
)
if not exist "%CONDA_PKGS_DIRS%" md "%CONDA_PKGS_DIRS%"
if not "%ERRORLEVEL%"=="0" (
  echo %ERROR% Failed to create directory "%CONDA_PKGS_DIRS%".
  set "_CACHE="
  exit /b !EXIT_STATUS!
)
set "PIP_CACHE_DIR=%_CACHE%\Python\pip"

echo %INFO% CACHE directory: "%_CACHE%".
echo %INFO% CONDA_PKGS_DIRS directory: "%CONDA_PKGS_DIRS%".
echo %INFO% PIP_CACHE_DIR   directory: "%PIP_CACHE_DIR%".

exit /b 0
:: ============================================================================
:: ============================================================================ CACHE_DIR END


:: ============================================================================ CHECK_DELAYED_EXPANSION BEGIN
:: ============================================================================
:CHECK_DELAYED_EXPANSION
::
:: Purpose:
::   Checks if Delayed Expansion is enabled.
::
:: Return:
::   DELAYED_EXPANSION=1 - Enabled
::   DELAYED_EXPANSION=0 - Disabled
::
:: Exit Codes:
::   0 - Enabled
::   1 - Disabled

echo:
echo %WARN% Checking cmd.exe delayed expansion availability
echo %INFO%
echo %INFO% When running with any arguments, Delayed Expansion feature must be
echo %INFO% enabled by the caller.
echo %INFO% When running withhout arguments, the script is supposed to spawn
echo %INFO% an activated shell, so Delayed Expansion can be enabled locally.
echo:
if "!ComSpec!"=="%ComSpec%" (
  set "DELAYED_EXPANSION=1"
  echo %INFO% --------------------------
  echo %OKOK% CHECK PASSED
  echo %INFO% Delayed Expansion enabled.
  echo %INFO% --------------------------
  echo:
  exit /b 0
) else (
  set "DELAYED_EXPANSION=0"
)

echo:
echo %INFO% ------------------------------------------------------------------------
echo %ERROR% CHECK FAILED
echo %WARN% Delayed Expansion disabled.
echo %INFO% This script should be generally called with Delayed Expansion enabled
echo %INFO% by the caller. When used interactively without any arguments, this 
echo %INFO% script will activate Conda environment aand spawn an activated shell.
echo %INFO% In this mode, Delayed Expansion mode can be activated by this script.
echo %INFO% In batch mode, this script is used to activate caller's environment,
echo %INFO% and, therefore this script will be unable to activate both Delayed
echo %INFO% Expansion and the caller's environment. Use one of the following
echo %INFO% options, then rerun this script with "/batch" switch to verify that the
echo %INFO% test passes.
echo %INFO% 
echo %INFO% 1. "setlocal EnableDelayedExpansion EnableExtensions"
echo %INFO%    Use this command in the parent script before calling this script.
echo %INFO%    
echo %INFO% 2. Start a new cmd.exe shell as "cmd.exe /E:ON /V:ON".
echo %INFO% 
echo %INFO% 3. Enable Delayed Expansion permanently via the following registry
echo %INFO%    setting (either variant should do), start a new shell, 
echo %INFO% ------------------------------------------------------------------------
echo: 
echo %INFO% Delayed expansion activation settings. 
echo: 
echo %INFO% ------------------------------------------------------------------------
echo %INFO% [HKEY_CURRENT_USER\Software\Microsoft\Command Processor]
echo %INFO% "DelayedExpansion"=dword:00000001
echo %INFO% "EnableExtensions"=dword:00000001
echo %INFO% 
echo %INFO% --- OR ---
echo %INFO% 
echo %INFO% [HKEY_LOCAL_MACHINE\Software\Microsoft\Command Processor]
echo %INFO% "DelayedExpansion"=dword:00000001
echo %INFO% "EnableExtensions"=dword:00000001
echo %INFO% ------------------------------------------------------------------------
echo:

exit /b 1
:: ============================================================================ 
:: ============================================================================ CHECK_DELAYED_EXPANSION END


:: ============================================================================ NO_ROOT_ENVIRONMENT BEGIN
:: ============================================================================
:: --------------------------------------------------------
:: NO_ROOT_ENVIRONMENT
::
:: This script should not be executed from a shell with active Python / Conda 
:: environment (visible conda.bat or python.exe). If found, issue a warning and
:: attempt to deactivate. Note, if active Python envirnoment was activated via
:: `conda activate` is should be possible to deactivate it via `conda deactivate`
:: If Python was placed on Path via Conda activation, deactivation should
:: remove it from Path. However, if Python is placed on Path independently,
:: for example via system-wide installation, deactivation will likely fail to
:: remove Python from Path. ALSO, if custom activation wrapper was used, such
:: as this very script, deactivation will not remove any custom envirnoment
:: settings. In such a case, effectively partial deactivation may result in
:: issues, potentially subtle, in the new environment. In particular, this
:: script activates external build dependencies for FFCV, and associated
:: settings may result in wrong dependency references and failed builds, if
:: new environment is not started from a clean system shell.
:: --------------------------------------------------------
:NO_ROOT_ENVIRONMENT

echo:
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN%         Checking for activated Conda environment           %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:

:: --- Check if Conda or Python is on the Path ---

set "_CONDA="
set "_PYTHON="
echo %INFO% Checking if Python on the Path
where "python.exe" 2>nul && (set "_PYTHON=python.exe")
echo %INFO% Checking if Conda on the Path
where "conda.bat" 2>nul && (set "_CONDA=conda.bat")
if not defined _CONDA if not defined _PYTHON (
  echo %OKOK% Python/Conda not detected.
  set "_CONDA="
  set "_PYTHON="
  exit /b 0
)

echo:
if defined _PYTHON (
  echo %WARN% Detected "python.exe" in Path.
)
if defined _CONDA (
  echo %WARN% Detected "conda.bat" in Path.
)

echo %WARN% It is strongly recommended to start this script from a clean
echo %WARN% environment. No Conda or Python variables should be in Path.

exit /b 1
:: ============================================================================ 
:: ============================================================================ NO_ROOT_ENVIRONMENT END
