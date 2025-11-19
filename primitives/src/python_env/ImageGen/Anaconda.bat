@echo off
setlocal EnableDelayedExpansion EnableExtensions

:: ============================================================================
::  Purpose:
::    Bootstraps a fully functional Conda/Micromamba-based Python environment.
::
::  Core YAML files:
::      - <script>_bootstrap.yml   - minimal bootstrap environment
::                                   Python/Conda/Mamba/UV
::      - <script>.yml             - main environment definition
::      - <script>_generated.yml   - generated full final resolved environment
::
::  Exit Codes:
::      0  - Success
::      1+ - Failure during environment or dependency setup
::
::  Related Scripts:
::      conda_far.bat   – Initializes and activates full Conda environment.
:: ============================================================================

echo :========== ========== ========== ========== ==========:
echo  Bootstrapping Python Environment
echo :---------- ---------- ---------- ---------- ----------:
rem           ----- Mon 10/20/2025 21:03:09.88 -----
echo:         ----- %DATE% %TIME% -----
echo: CLI: "%~f0" %*
echo:

:: --- Parse arguments and preserve top-level context ---

call :PARSE_ARGS %*

:: --- Escape sequence templates for color coded console output ---

call :COLOR_SCHEME

:: --- Determine base components of environment path and check for existing Python ---

set "_ENV_PREFIX=%~dpn0"
if exist "%_ENV_PREFIX%\python.exe" (
  echo %WARN% Found existing "%_ENV_PREFIX%\python.exe". Skip bootstrapping...
  goto :CLEANUP
)

:: --- Check prerequisites ---

call :PREREQUISITES
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed prerequisite checks. See error above. Aborting...
  goto :CLEANUP
)

:: --------------------------------------------------------
:: BASE CONFIG
:: --------------------------------------------------------
set "YAML_BOOTSTRAP=%~dpn0_bootstrap.yml"
if not exist "!YAML_BOOTSTRAP!" (
  echo %ERROR% Bootstrap environment file "!YAML_BOOTSTRAP!" not found. Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
echo %INFO% Using bootstrap environment file "!YAML_BOOTSTRAP!".

set "YAML=%~dpn0.yml"
if not exist "!YAML!" (
  echo %ERROR% Main environment file "!YAML!" not found. Aborting...
  set "FINAL_EXIT_CODE=1"
  goto :CLEANUP
)
echo %INFO% Using bootstrap environment file "!YAML!".

:: --------------------------------------------------------
:: VERBOSITY
:: --------------------------------------------------------
if not defined _ARG_Q (
  set "VERBOSE=-v"
) else (
  set "VERBOSE="
)

:: --------------------------------------------------------
:: Determine cache directory
:: --------------------------------------------------------
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
pause

:: --------------------------------------------------------
:: Download Micromamba
:: --------------------------------------------------------
call :MICROMAMBA_DOWNLOAD
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed to download/verify Micromamba. Aborting...
  goto :CLEANUP
)

:: --------------------------------------------------------
:: Bootstrap new Python/Conda/Mamba/UV environment
:: --------------------------------------------------------
call :BOOTSRTAP_ENV
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed to bootsrap Python/Conda/Mamba/UV environment. Aborting...
  goto :CLEANUP
)

:: --------------------------------------------------------
:: Activate new Python/Conda/Mamba/UV environment
:: --------------------------------------------------------
call :ACTIVATE_ENV
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed to activate the new Python/Conda/Mamba/UV environment. Aborting...
  goto :CLEANUP
)
if defined _ARG_INIT (
  echo %INFO% Init-only flag supplied. Exiting...
  set "FINAL_EXIT_CODE=0"
  goto :CLEANUP
)

:: --------------------------------------------------------
:: Import main environment
:: --------------------------------------------------------
call :IMPORT_MAIN_ENV
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed to import main environment. Aborting...
  goto :CLEANUP
)

:: --------------------------------------------------------
:: Export full environment
:: --------------------------------------------------------
call :EXPORT_FULL_ENV
if not "%ERRORLEVEL%"=="0" (
  set "FINAL_EXIT_CODE=%ERRORLEVEL%"
  echo %ERROR% Failed to export full environment. Aborting...
  goto :CLEANUP
)

echo:
rem                   ----- Mon 10/20/2025 21:03:09.88 -----
echo:                 ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %OKOK%                                                            %OKOK% ==
echo == %OKOK%   ImageGen environment created and verified successfully.  %OKOK% ==
echo == %OKOK%                                                            %OKOK% ==
echo ====================================================================================
echo ====================================================================================
echo:

set "FINAL_EXIT_CODE=0"
goto :CLEANUP
:: ============================================================================
:: ============================================================================
:: ============================================================================


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

set "_ARG_INIT="
set "_ARG_NOCOLOR="
set "_ARG_Q="

:PARSE_NEXT_ARG

if /I "%~1"=="" goto :PARSE_ARGS_DONE
if /I "%~1"=="/q"         set "_ARG_Q=1"
if /I "%~1"=="/init"      set "_ARG_INIT=1"
if /I "%~1"=="/nocolor"   set "_ARG_NOCOLOR=1"
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

set "_ARG_INIT="
set "_ARG_NOCOLOR="
set "_ARG_Q="

:: --- Ensure a valid exit code is always returned ---

if not defined FINAL_EXIT_CODE set "FINAL_EXIT_CODE=1"
exit /b %FINAL_EXIT_CODE%
:: ============================================================================ 
:: ============================================================================ CLEANUP END


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


:: ============================================================================ MICROMAMBA_DOWNLOAD BEGIN
:: ============================================================================
:MICROMAMBA_DOWNLOAD

:: --------------------------------------------------------
:: Download Micromamba
:: --------------------------------------------------------
echo %WARN% Micromamba
set "RELEASE_URL=https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64"
set "MAMBA_EXE=%_CACHE%\micromamba\micromamba.exe"

if exist "%MAMBA_EXE%" (
  echo %INFO% Micromamba: Using cached "%MAMBA_EXE%"
  goto :MICROMAMBA_DOWNLOAD_SKIP
) 

if not exist "%_CACHE%\micromamba" md "%_CACHE%\micromamba"
set "EXIT_STATUS=!ERRORLEVEL!"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Creating "%_CACHE%\micromamba". Aborting...
  exit /b %EXIT_STATUS%
)

echo %INFO% Micromamba: Downloading: %RELEASE_URL%
echo %INFO% Micromamba: Destination: %MAMBA_EXE%
curl --fail --retry 3 --retry-delay 2 -L -o "%MAMBA_EXE%" "%RELEASE_URL%"
set "EXIT_STATUS=!ERRORLEVEL!"
if not "!EXIT_STATUS!"=="0" (
  echo %ERROR% Micromamba: Download failure. Aborting bootstrapping...
  exit /b !EXIT_STATUS!
)

set "RELEASE_URL="
if not exist "%MAMBA_EXE%" (
  echo %ERROR% Micromamba: File "%MAMBA_EXE%" missing after download. Aborting...
  exit /b 1
)

:MICROMAMBA_DOWNLOAD_SKIP

echo %INFO% Micromamba: Basic test of "%MAMBA_EXE%": Attempt to query Micromamba version.
"%MAMBA_EXE%" --version
set "EXIT_STATUS=%ERRORLEVEL%"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failure to execute "%MAMBA_EXE%". ERRORLEVEL: "%EXIT_STATUS%". Aborting...
  exit /b %EXIT_STATUS%
)

echo %OKOK% Micromamba: Completed
echo:

exit /b 0
:: ============================================================================
:: ============================================================================ MICROMAMBA_DOWNLOAD END


:: ============================================================================ BOOTSRTAP_ENV BEGIN
:: ============================================================================
:BOOTSRTAP_ENV

:: --------------------------------------------------------
:: Bootstrap new Python/Conda/Mamba/UV environment
:: --------------------------------------------------------
echo:
rem                   ----- Mon 10/20/2025 21:03:09.88 -----
echo:                 ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN%           Bootstrapping new Python environment             %WARN% ==
echo == %WARN%           Python/Conda/Mamba/UV                            %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:
rem set "PKGS=mamba conda uv %_PYTHON_PKG%"

if exist "%APPDATA%\mamba" (
  echo %WARN% Warning: I am about to delete "%APPDATA%\mamba". Press any key to continue.
  echo %WARN% Somehow, Micromamba tends to hang when this directory exists.
  pause
  rmdir /Q /S "%APPDATA%\mamba"
  set "EXIT_STATUS=!ERRORLEVEL!"
  if not "!EXIT_STATUS!"=="0" (
    echo %ERROR% Failed to delete "%APPDATA%\mamba". ERRORLEVEL: %EXIT_STATUS%. Aborting...
    exit /b %EXIT_STATUS%
  )
)

echo %WARN% Creating new Python environment...
echo %INFO% Using command:
echo %INFO% === "%MAMBA_EXE%" create %VERBOSE% --yes --no-rc --use-uv -f "%YAML_BOOTSTRAP%" --prefix "%_ENV_PREFIX%" %PKGS% ===
echo %INFO%
echo:
call "%MAMBA_EXE%" create %VERBOSE% --yes --no-rc --use-uv -f "%YAML_BOOTSTRAP%" --prefix "%_ENV_PREFIX%" %PKGS%
set "EXIT_STATUS=%ERRORLEVEL%"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to create a new environment. ERRORLEVEL: %EXIT_STATUS%. Aborting...
  exit /b %EXIT_STATUS%
)

echo %OKOK% New environment "%_ENV_PREFIX%" is bootstrapped from "%YAML_BOOTSTRAP%".
exit /b 0
:: ============================================================================
:: ============================================================================ BOOTSRTAP_ENV END


:: ============================================================================ ACTIVATE_ENV BEGIN
:: ============================================================================
:ACTIVATE_ENV

echo:
rem                   ----- Mon 10/20/2025 21:03:09.88 -----
echo:                 ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN%            Activate development environment.               %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:

set "_CONDA_PREFIX=%_ENV_PREFIX%"
call "%~dp0conda_far.bat" /preactivate
set "EXIT_STATUS=!ERRORLEVEL!"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to activate environment "%_ENV_PREFIX%". Aborting...
  exit /b %EXIT_STATUS%
)
set "_CONDA_PREFIX="

if not exist "%CONDA_PREFIX%\python.exe" (
  echo %ERROR% Python not found in "%CONDA_PREFIX%". Aborting...
  exit /b 1
)
call "%CONDA_PREFIX%\python.exe" --version
set "EXIT_STATUS=!ERRORLEVEL!"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to call Python in "%_ENV_PREFIX%". Aborting...
  exit /b %EXIT_STATUS%
)
call :COLOR_SCHEME

echo %OKOK% New environment "%_ENV_PREFIX%" is activated.
exit /b 0
:: ============================================================================
:: ============================================================================ ACTIVATE_ENV END


:: ============================================================================ IMPORT_MAIN_ENV BEGIN
:: ============================================================================
:IMPORT_MAIN_ENV
:: --------------------------------------------------------
:: Import main environment
:: --------------------------------------------------------
echo:
rem                   ----- Mon 10/20/2025 21:03:09.88 -----
echo:                 ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN%           Importing main Python environment                %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:
echo %INFO% YAML:   "%YAML%"
echo %INFO% PREFIX: "%CONDA_PREFIX%"
echo %INFO%

call "%MAMBA_BAT%" env update %VERBOSE% --yes --no-rc --use-uv -f "%YAML%" --prefix "%CONDA_PREFIX%"
set "EXIT_STATUS=!ERRORLEVEL!"
echo:
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to import main environment "%_ENV_PREFIX%". Aborting...
  exit /b %EXIT_STATUS%
)

echo %OKOK% Imported main environment "%YAML%" to "%_ENV_PREFIX%".
exit /b 0
:: ============================================================================
:: ============================================================================ IMPORT_MAIN_ENV END


:: ============================================================================ EXPORT_FULL_ENV BEGIN
:: ============================================================================
:EXPORT_FULL_ENV
echo:
rem                   ----- Mon 10/20/2025 21:03:09.88 -----
echo:                 ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN%           Exporting final full environment file            %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:
echo %INFO% Exporting final full environment file to "%YAML:.yml=_generated.yml%".

call "%CONDA_BAT%" env export --no-builds > "%YAML:.yml=_generated.yml%"
set "EXIT_STATUS=!ERRORLEVEL!"
if not "!EXIT_STATUS!"=="0" (
  echo %ERROR% Failed to export environment file. Aborting...
  exit /b %EXIT_STATUS%
)

echo %OKOK% Exported full environment to "%YAML:.yml=_generated.yml%".
exit /b 0
:: ============================================================================
:: ============================================================================ EXPORT_FULL_ENV END


:: ============================================================================ EXTRA_ENV BEGIN
:: ============================================================================
:EXTRA_ENV

:: --- Not currently used. Keeping for now for potential use. ---

setlocal
:: --------------------------------------------------------
:: Import additional environment
:: --------------------------------------------------------
echo:
echo %INFO% EXTRA ENVIRONMENTS

:: --------------------------------------------------------
:: Activate environment
:: --------------------------------------------------------
if defined CONDA_PREFIX goto :SKIP_EXTRA_ACTIVATION
call "%~dp0conda_far.bat" /batch
set "EXIT_STATUS=!ERRORLEVEL!"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to activate environment "%_ENV_PREFIX%". Aborting...
  exit /b %EXIT_STATUS%
)
call python.exe --version
set "EXIT_STATUS=!ERRORLEVEL!"
if not "%EXIT_STATUS%"=="0" (
  echo %ERROR% Failed to call Python in "%_ENV_PREFIX%". Aborting...
  exit /b %EXIT_STATUS%
)
:SKIP_EXTRA_ACTIVATION

:: --------------------------------------------------------
:: Command line environment
:: --------------------------------------------------------
echo:
echo %INFO% CHECKING COMMAND LINE EXTRA ENVIRONMENT

set "YAML_EXTRA=%~1"
if not defined YAML_EXTRA goto :SKIP_ARG_YAML
if /I not "%YAML_EXTRA:~-4%"==".yml" (
  echo %WARN% First argument is not *.yml: "%YAML_EXTRA%". Skipping...
  goto :SKIP_ARG_YAML
)
if not exist "%YAML_EXTRA%" (
  echo %WARN% "%YAML_EXTRA%" does not exist. Skipping...
  goto :SKIP_ARG_YAML
)

echo %INFO% importing environment file "%YAML_EXTRA%".
rem mamba hangs with Python 3.9 (not clear if the old Python version causes specific problems here)?
rem call "%MAMBA_BAT%" env update -vv --yes --no-rc --use-uv -f "%YAML_EXTRA%" --prefix "%_ENV_PREFIX%"

call "%CONDA_BAT%" env update -v -f "%YAML_EXTRA%" --prefix "%_ENV_PREFIX%"

:SKIP_ARG_YAML
:: --------------------------------------------------------
:: Default extra environment
:: --------------------------------------------------------
echo:
echo %INFO% CHECKING DEFAULT EXTRA ENVIRONMENT

set "YAML_EXTRA=%YAML:~0,-4%_Extra_Default.yml"
if exist "%YAML_EXTRA%" (
  echo %INFO% Found default extra env file "%YAML_EXTRA%". Importing...
  call "%CONDA_BAT%" env update -v -f "%YAML_EXTRA%" --prefix "%_ENV_PREFIX%"
)

endlocal & exit /b 0
:: ============================================================================
:: ============================================================================ EXTRA_ENV END


:: ============================================================================ PREREQUISITES BEGIN
:: ============================================================================
:: --------------------------------------------------------
:: CHECK Prerequisites
:: --------------------------------------------------------
:PREREQUISITES

rem                       ----- Mon 10/20/2025 21:03:09.88 -----
echo:                     ----- %DATE% %TIME% -----
echo ====================================================================================
echo ====================================================================================
echo == %WARN%                                                            %WARN% ==
echo == %WARN% PREREQS: Checking prerequisites.                           %WARN% ==
echo == %WARN% PREREQS: Inspect results and make sure that all tests are  %WARN% ==
echo == %WARN%          OK and no ERRORs reported before continuing.      %WARN% ==
echo == %WARN%                                                            %WARN% ==
echo ====================================================================================
echo ====================================================================================
echo:

:: --------------------------------------------------------
:: Required scripts
:: --------------------------------------------------------
echo %WARN% PREREQS - Scripts

:: --- conda_far.bat ---

if exist "%~dp0conda_far.bat" (
  echo %OKOK% PREREQS - Scripts: Conda wrapper script found: "%~dp0conda_far.bat". 
) else (
  echo %ERROR% PREREQS - Scripts: Conda wrapper script not found: "%~dp0conda_far.bat". Aborting...
  exit /b 1
)
echo:

exit /b 0
:: ============================================================================ 
:: ============================================================================ PREREQUISITES END
