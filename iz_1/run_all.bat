@echo off
REM Пример пакетного запуска: поменяйте список файлов под себя
setlocal enabledelayedexpansion

set VIDS= v1.mp4 v2.mp4 v3.mp4 v4.mp4 v5.mp4
set METHODS= csrt kcf mosse ncc

for %%V in (%VIDS%) do (
  for %%M in (%METHODS%) do (
    echo Running %%M on %%V ...
    python trackers\runner.py --method %%M --video data\%%V --save out\%%~nV_%%M.mp4
  )
)

echo Собираем метрики...
python metrics\eval.py --logs logs --out metrics_summary.csv
echo Done.
