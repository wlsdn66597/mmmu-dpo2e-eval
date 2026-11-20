#!/usr/bin/env bash
# DPO2-E 모델을 E와 같은 하네스·설정으로 MMMU test(10,500문항)에서 평가한다. 생성 seed만 바꾼다.
#   nohup bash run_test.sh <DPO2E_repro_adapter_eval.zip> <seed> > test_<seed>.log 2>&1 &
# setup.sh를 먼저 끝내야 한다. 끊기면 같은 명령으로 다시 실행하면 이어서 한다.
# 결과: harness_split/outputs/dpo2E_test_seed<seed>/ (scores.md, summary.json, predictions.jsonl, results.json)
set -euo pipefail
if [ $# -lt 2 ]; then sed -n '2,5p' "$0"; exit 1; fi
ZIP=$(readlink -f "$1"); SEED=$2
cd "$(dirname "$0")"; ROOT=$PWD
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed는 정수여야 한다: $SEED"; exit 1; }
[ -f "$ZIP" ] || { echo "zip이 없다: $ZIP"; exit 1; }
for f in .venv/.deps_ok harness_split/.patched harness_split/outputs/preparation.json; do
  [ -f "$f" ] || { echo "setup.sh가 끝나지 않았다 ($f 없음)"; exit 1; }
done
PY=$ROOT/.venv/bin/python
RUN=dpo2E_test_seed$SEED
CFG=$ROOT/harness_split/src/config_test_seed$SEED.json
OUTDIR=$ROOT/harness_split/outputs/$RUN
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
log "seed $SEED, run $RUN"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# 1. zip 풀기
if [ ! -f dpo2e/adapter/adapter_model.safetensors ] || [ ! -f dpo2e/adapter/gates.json ]; then
  log "1. zip 풀기"
  rm -rf dpo2e; "$PY" -I -m zipfile -e "$ZIP" dpo2e
fi

# 2. 병합 + 게이트 흡수 (CPU). E가 val 평가에 쓴 병합 모델과 SHA-256 비교
if [ ! -f dpo2e/merged/.merge_ok ]; then
  log "2. 병합"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" "$PY" -I scripts/merge_dpo2e.py dpo2e/adapter dpo2e/eval_mmmu_val/manifest.json dpo2e/merged
fi
cat dpo2e/merged/.merge_ok

# 3. 평가: 생성(1문항씩) → Qwen3-8B 판정 → 채점
"$PY" -I scripts/make_config.py harness_split/src/config.json "$CFG" "$SEED"
for try in 1 2 3 4 5 6; do
  [ -f "$OUTDIR/summary.json" ] && break
  for f in "$OUTDIR/predictions.jsonl" "$OUTDIR/judge_attempts.jsonl"; do
    [ -s "$f" ] && "$PY" -I scripts/trim_partial.py "$f"
  done
  log "3. 평가 (시도 $try, 저장된 응답 $( [ -f "$OUTDIR/predictions.jsonl" ] && wc -l < "$OUTDIR/predictions.jsonl" || echo 0 )/10500)"
  if "$PY" harness_split/src/evaluate.py --config "$CFG" --checkpoint "$ROOT/dpo2e/merged" --run-name "$RUN"; then break; fi
  log "평가가 중단됨. 60초 뒤 이어서 한다"; sleep 60
done

if [ -f "$OUTDIR/summary.json" ]; then
  log "끝. 결과 $OUTDIR"; cat "$OUTDIR/scores.md"
else
  log "6번 시도했지만 끝나지 않았다. 위 오류를 확인한 뒤 같은 명령으로 다시 실행한다"; exit 1
fi
