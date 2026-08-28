#!/bin/bash
# nohup wrapper: run baseline grid -> plot -> write DONE
cd "d:/notes/Ablaze/pages/理工/计算机/申请导师快速练习项目/qaoa-repro" || exit 1
PY="E:/software/miniforge/python.exe"

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] baseline grid start ==="
"$PY" run_baseline.py > run.log 2>&1
BASE_EXIT=$?

if [ "$BASE_EXIT" -ne 0 ]; then
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] BASELINE FAILED (exit=$BASE_EXIT) ===" >> run.log
  echo "BASELINE_FAILED exit=$BASE_EXIT" > DONE
  exit 1
fi

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] plotting ==="
"$PY" plot_results.py >> run.log 2>&1
PLOT_EXIT=$?

if [ "$PLOT_EXIT" -ne 0 ]; then
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] PLOT FAILED (exit=$PLOT_EXIT) ===" >> run.log
  echo "PLOT_FAILED exit=$PLOT_EXIT" > DONE
  exit 1
fi

echo "ALL_DONE" > DONE
echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] all done ===" >> run.log
