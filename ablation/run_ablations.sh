#!/usr/bin/env bash
# E recipe ablation 3개를 차례로 학습 → 병합 → MMMU val(seed 3407, 과제 harness) 평가한다.
#   1 prefix    : rejected 점수를 loop onset이 아니라 응답 처음부터 계산 (나머지 E와 동일)
#   2 mainonly  : loop 쌍 ×4 + NLL만 (ERA 쌍·replay·시각 연결부 LoRA·gate·이미지 조건 손실 제거)
#   3 novision  : E에서 시각 연결부 LoRA·gate·이미지 조건 손실만 제거
#
# 사용 (repo 폴더에서, setup.sh와 run_test.sh 한 번 이상 실행해 dpo2e/ 가 풀려 있어야 함):
#   setsid nohup bash ablation/run_ablations.sh <pairs_E.jsonl> <이미지 루트> > ablation.log 2>&1 < /dev/null &
#   - 팀원 묶음(pairs_E_bundle.tar)을 ~/pairsE 에 풀었다면: ~/pairsE/pairs_E.jsonl ~/pairsE/vic_era_2000
#   - 쌍 파일은 E 재현 학습과 같은 것(2,307행, md5 9b7b6d94…)이어야 한다. 다르면 멈춘다 (FORCE=1 로 무시).
#   - 일부만: ONLY="prefix novision" bash ablation/run_ablations.sh …
# 끊기면 같은 명령으로 다시 실행: 병합까지 끝난 학습과 채점까지 끝난 평가는 건너뛴다.
# 결과: ablation_work/ablation_summary.md (E 재현 val 602와 나란히 표), harness_split/outputs/abl_<이름>_val_seed3407/
set -euo pipefail
if [ $# -lt 2 ]; then sed -n '2,13p' "$0"; exit 1; fi
cd "$(dirname "$0")/.."; ROOT=$PWD
PAIRS=$(readlink -f "$1"); IMAGES=$(readlink -f "$2")
W=$ROOT/ablation_work; mkdir -p "$W"
PY=$ROOT/.venv/bin/python
CFG=$ROOT/harness_split/src/config_val_seed3407.json
JOBS=${ONLY:-"prefix mainonly novision"}
export CC=${CC:-$HOME/bin/cc} HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 PATH="$HOME/.local/bin:$PATH"
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for f in .venv/.deps_ok harness_split/.patched dpo2e/code/train_dpo_d.py dpo2e/logs/train_info.json; do
  [ -e "$f" ] || { log "준비 안 됨: $f 없음 (setup.sh, run_test.sh를 먼저 실행)"; exit 1; }
done

# ---------------------------------------------------------------- 0. 준비
log "0. 입력 검사·코드 준비"
"$PY" -I ablation/prep.py "$ROOT" "$PAIRS" "$IMAGES" "$W" $([ "${FORCE:-0}" = 1 ] && echo --force)
"$PY" -I scripts/make_config.py harness_split/src/config.json "$CFG" 3407 validation
if [ ! -f harness_split/outputs/.data_val_ok ]; then
  if ! (cd harness_split/src && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "$PY" -c       "import prepare; c=prepare.config('$CFG'); prepare.load_data(c, prepare.cache_paths(c)['dataset'])" >/dev/null 2>&1); then
    log "0. MMMU val 데이터 받기"
    (cd harness_split/src && "$PY" -c       "import prepare; c=prepare.config('$CFG'); prepare.load_data(c, prepare.cache_paths(c)['dataset'], offline=False)")
  fi
  touch harness_split/outputs/.data_val_ok
fi
BASE=$(cd harness_split/src && HF_HUB_OFFLINE=1 "$PY" -c "import prepare; c=prepare.config('$CFG'); print(prepare.snapshot(c, prepare.cache_paths(c)['model']))" | tail -1)
[ -f "$BASE/config.json" ] || { log "base 모델을 찾지 못함: $BASE"; exit 1; }
log "base 모델: $BASE"

# 학습은 평가 가상환경(torch 2.8.0·transformers 4.57.0·peft 0.17.1 = E 학습 버전)을 그대로 쓰고,
# flash_attn만 별도 폴더에 받아 학습 때만 PYTHONPATH로 붙인다 (평가 환경 자체는 바꾸지 않는다)
LIB=$W/pylib
if [ ! -f "$LIB/.ok" ]; then
  log "0. flash_attn 2.8.3 받기 (학습 전용 폴더, 약 250MB)"
  rm -rf "$LIB"
  if "$PY" -m pip install -q --no-deps --target "$LIB"       "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl#sha256=f25da18657a87fc83dc1bfb8b7751b82246e9db355510226b674fd437c34b5fb"; then
    touch "$LIB/.ok"
  else log "flash_attn 설치 실패"; fi
fi
if PYTHONPATH=$LIB "$PY" -c "import flash_attn" 2>/dev/null; then ATTN=flash_attention_2; else ATTN=sdpa; fi
log "학습 attention 구현: $ATTN (E 학습은 flash_attention_2)"

wait_gpu() {
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 2000 ] && ! pgrep -f "evaluate.py|run_mmmu_pro|train_dpo_d.py" >/dev/null; then return; fi
    log "GPU 사용 중(${used}MiB) → 5분 뒤 다시 확인"; sleep 300
  done
}

COMMON="--beta 0.1 --rpo-alpha 1.0 --epochs 1 --grad-accum 16 --lr 1e-5 --warmup-ratio 0.03 --weight-decay 0.01
 --max-grad-norm 1.0 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --max-tokens 8400 --seed 3407 --workers 2
 --gate-lr 1e-3 --img-pref-weight 1.0 --replay-weight 1.0"

train() {  # name pairs from_start extra-flags...
  local name=$1 pairs=$2 from_start=$3; shift 3
  local out=$W/$name/train
  if [ -f "$out/merged/config.json" ] && ls "$out"/merged/*.safetensors >/dev/null 2>&1; then
    log "$name: 학습·병합 끝남, 건너뜀"; return
  fi
  rm -rf "$out"; mkdir -p "$W/$name"
  wait_gpu
  log "$name: 학습 시작 (쌍 $(wc -l < "$pairs")행, from_start=$from_start, 추가 옵션: $*)"
  if ! ABL_SCORE_FROM_START=$from_start ABL_ATTN=$ATTN PYTHONPATH=$LIB "$PY" "$W/code/train_dpo_d.py" --pairs "$pairs" --images "$IMAGES" \
    --start-model "$BASE" --vision-config "$W/code/training_settings_only.json" --diag "$W/code" --out "$out" \
    $COMMON "$@" > "$W/$name/train.log" 2>&1; then
    log "$name: 학습 실패 → 이 변형은 건너뜀 ($W/$name/train.log 끝부분 확인)"; tail -5 "$W/$name/train.log"; return 1
  fi
  log "$name: 학습·병합 완료 ($(tail -1 "$out/train_log.jsonl" | cut -c1-160))"
}

evaluate() {  # name
  local name=$1 run=abl_$1_val_seed3407
  local o=$ROOT/harness_split/outputs/$run
  for try in 1 2 3 4; do
    [ -f "$o/summary.json" ] && break
    for f in "$o/predictions.jsonl" "$o/judge_attempts.jsonl"; do [ -s "$f" ] && "$PY" -I scripts/trim_partial.py "$f"; done
    wait_gpu
    log "$name: val 평가 (시도 $try, 저장된 응답 $( [ -f "$o/predictions.jsonl" ] && wc -l < "$o/predictions.jsonl" || echo 0 )/900)"
    if "$PY" harness_split/src/evaluate.py --config "$CFG" --checkpoint "$W/$name/train/merged" --run-name "$run" \
         >> "$W/$name/eval.log" 2>&1; then break; fi
    log "$name: 평가 중단, 60초 뒤 이어서"; sleep 60
  done
  [ -f "$o/summary.json" ] && log "$name: $(tail -1 "$o/scores.md")" || log "$name: 평가 미완료 ($W/$name/eval.log 확인)"
}

for job in $JOBS; do
  case $job in
    prefix)   train prefix   "$W/pairs_full.jsonl"      1 --vision-lora --gates --img-pref-types V,I && evaluate prefix || true ;;
    mainonly) train mainonly "$W/pairs_loop_only.jsonl" 0 && evaluate mainonly || true ;;
    novision) train novision "$W/pairs_full.jsonl"      0 && evaluate novision || true ;;
    *) log "알 수 없는 job: $job" ;;
  esac
done

"$PY" -I ablation/summarize.py dpo2e/eval_mmmu_val \
  $(for j in $JOBS; do echo "harness_split/outputs/abl_${j}_val_seed3407"; done) > "$W/ablation_summary.md"
log "모두 끝. 표: $W/ablation_summary.md"; cat "$W/ablation_summary.md"
