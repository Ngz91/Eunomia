@echo off
if "%1" == "-s" (
python path\to\eunomia.py %1
)
if "%1" == "start" (
python path\to\Eunomia.py %1
)
if "%1" == "-i" (
python path\to\eunomia.py %1
)
if "%1" == "ingest" (
python path\to\eunomia.py %1
)
if "%1" == "" (
echo "No argument"
)