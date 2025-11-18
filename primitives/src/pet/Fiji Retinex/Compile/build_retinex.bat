@echo off
setlocal


set VERSION=1.54p
set FIJI_DIR=G:\ProgramsMisc\Fiji
set JAVABIN=%FIJI_DIR%\java\win64\zulu8.86.0.25-ca-fx-jdk8.0.452-win_x64\bin

echo ---------------------------------------
echo Building Retinex_.jar for Fiji 1.54p...
echo ---------------------------------------

REM --- Locate Fiji installation automatically ---
if not exist "%FIJI_DIR%" (
    echo ERROR: Fiji not found
    echo Edit build_retinex.bat and set FIJI_DIR manually.
    pause
    exit /b 1
)

REM --- Locate Fiji's ImageJ 1.x API ---
set IJ_CLASSPATH="%FIJI_DIR%\jars\ij-%VERSION%.jar"

if not exist %IJ_CLASSPATH% (
    echo ERROR: Cannot find ij.jar in %FIJI_DIR%\jars
    pause
    exit /b 1
)

REM --- Compile plugin using Java 8 toolchain built into Fiji ---
set JAVAC="%JAVABIN%\javac.exe"
if not exist %JAVAC% (
    echo ERROR: Java compiler not found inside Fiji.
    echo Install Adoptium JDK 8 or JDK 11.
    pause
    exit /b 1
)

%JAVAC% -source 1.8 -target 1.8 -classpath %IJ_CLASSPATH% Retinex_.java

if errorlevel 1 (
    echo Compilation failed.
    pause
    exit /b 1
)

echo ---------------------------------------
echo Creating Retinex_.jar...
echo ---------------------------------------

REM --- Package .class into a jar ---
"%JAVABIN%\jar.exe" cvf Retinex_.jar Retinex_.class

echo.
echo SUCCESS!
echo Your plugin is ready:
echo   Retinex_.jar
echo.
echo Copy it into:
echo   %FIJI_DIR%\plugins\
echo Then restart Fiji.
echo ---------------------------------------

pause
