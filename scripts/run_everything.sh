#!/usr/bin/env bash
# Runs every executable stage of the project, in dependency order, in one pass.
# Not part of the delivered code path (that's Makefile / individual scripts) -
# this exists purely to produce one consolidated, timestamped proof-of-execution
# log for the FYP write-up. Continues past a failing stage so the log shows the
# full picture rather than stopping at the first problem.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

LOG=reports/RUN_LOG.md
STAGE_FAILS=0

echo "# Full project run log" > "$LOG"
echo "" >> "$LOG"
echo "Generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG"
echo "" >> "$LOG"
echo "| # | stage | command | result | duration |" >> "$LOG"
echo "|---|---|---|---|---|" >> "$LOG"

run_stage () {
  local n="$1" name="$2" cmd="$3"
  echo ""
  echo "================================================================"
  echo "[$n] $name"
  echo "    $ $cmd"
  echo "================================================================"
  local start end dur status
  start=$(date +%s)
  eval "$cmd" > "/tmp/stage_${n}.log" 2>&1
  status=$?
  end=$(date +%s)
  dur=$((end - start))
  if [ $status -eq 0 ]; then
    echo "[$n] $name -> OK (${dur}s)"
    echo "| $n | $name | \`$cmd\` | OK | ${dur}s |" >> "$LOG"
  else
    echo "[$n] $name -> FAILED (exit $status, ${dur}s) - see /tmp/stage_${n}.log"
    tail -30 "/tmp/stage_${n}.log"
    echo "| $n | $name | \`$cmd\` | **FAILED (exit $status)** | ${dur}s |" >> "$LOG"
    STAGE_FAILS=$((STAGE_FAILS + 1))
  fi
  tail -15 "/tmp/stage_${n}.log"
}

run_stage 1  "data validation + 70/15/15 split"        "python scripts/prepare_datasets.py"
run_stage 2  "forecast evaluation (baselines)"          "python scripts/run_forecast_eval.py"
run_stage 3  "backend unit tests (38)"                  "python -m unittest discover -s backend/tests -p 'test_*.py'"
run_stage 4  "ML/SMOTE unit tests (20)"                 "python -m unittest discover -s ml/tests -p 'test_*.py'"
run_stage 5  "MongoDB repository/security tests (15)"   "python -m unittest discover -s mongo/tests -p 'test_*.py'"
run_stage 6  "embedding model selection (RAG-style)"    "python -m augmentation.model_selection"
run_stage 7  "retrieval-grounded augmentation (~2M rows)" "python -m augmentation.run_augmentation"
run_stage 8  "6-model comparison + SHAP + charts"       "python scripts/run_model_comparison.py"
run_stage 9  "Pipeline A: training + registry + seed"   "python pipelines/training_pipeline.py --dataset both"
run_stage 10 "Pipeline B: alerting + restock (empty DB, standalone process)" "python pipelines/automation_pipeline.py --once"
run_stage 11 "Pipeline A -> B chained (shared in-memory DB, realistic flow)" "python scripts/demo_pipelines_end_to_end.py"

echo "" >> "$LOG"
if [ $STAGE_FAILS -eq 0 ]; then
  echo "**All 11 stages completed successfully.**" >> "$LOG"
else
  echo "**$STAGE_FAILS stage(s) failed - see logs above.**" >> "$LOG"
fi

echo ""
echo "================================================================"
echo "DONE. $STAGE_FAILS stage(s) failed. Full log: $LOG"
echo "================================================================"
cat "$LOG"
exit $STAGE_FAILS
