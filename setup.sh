#!/usr/bin/env bash
# 새 컴퓨터에서 한 번 실행한다: Python 3.12 가상환경, 고정 패키지, 평가 하네스, 모델·데이터 다운로드.
#   bash setup.sh
# 중간에 끊기면 다시 실행하면 끝난 단계는 건너뛴다.
set -euo pipefail
cd "$(dirname "$0")"; ROOT=$PWD

HARNESS_REPO=https://github.com/wlsdn66597/mmdl-baseline.git
HARNESS_COMMIT=7740786a1b6b173db86285a00f06e93a44d8a225
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
export GIT_TERMINAL_PROMPT=0  # 백그라운드에서 git이 입력을 기다리며 멈추지 않고 바로 오류를 내도록
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"  # uv 설치 직후 셸을 새로 열지 않아도 찾도록
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for c in git nvidia-smi; do command -v "$c" >/dev/null || { log "$c 명령이 없다"; exit 1; }; done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

# 1. Python 3.12 가상환경
if [ ! -x .venv/bin/python ]; then
  log "1. Python 3.12 가상환경 만들기"
  if command -v python3.12 >/dev/null && python3.12 -m venv .venv; then :
  elif command -v uv >/dev/null; then rm -rf .venv; uv venv -q --python 3.12 --seed .venv
  else rm -rf .venv; log "Python 3.12이 필요하다. sudo 없이: curl -LsSf https://astral.sh/uv/install.sh | sh 후 다시 실행"; exit 1
  fi
fi
PY=$ROOT/.venv/bin/python
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# 2. 패키지 (평가 하네스의 고정 목록 + 병합용 peft·accelerate. 고정 목록의 버전은 바꾸지 않는다)
if [ ! -f .venv/.deps_ok ]; then
  log "2. 패키지 설치"
  "$PY" -m pip install -q -r env/requirements.txt
  "$PY" -m pip install -q -c env/requirements.txt -r env/requirements_merge.txt
  touch .venv/.deps_ok
fi

# 3. 평가 하네스 (E가 MMMU-val을 평가한 코드 그대로)
if [ ! -d harness/.git ]; then
  log "3. 평가 하네스 받기"
  git clone -q "$HARNESS_REPO" harness
fi
git -C harness checkout -q "$HARNESS_COMMIT"
"$PY" -I scripts/check_harness.py harness

# 4. test용 하네스 복사본 (바꾼 곳은 harness_split/PATCH_NOTES.md)
if [ ! -f harness_split/.patched ]; then
  log "4. test용 하네스 복사본 만들기"
  rm -rf harness_split; cp -a harness harness_split
  "$PY" -I scripts/patch_harness.py harness_split
  touch harness_split/.patched
fi
"$PY" -I scripts/make_config.py harness_split/src/config.json harness_split/src/config_test_seed3407.json 3407

# 5. 모델·데이터 (하네스 prepare.py가 고정 revision으로 받는다: Qwen3-VL-4B-Instruct, Qwen3-8B, MMMU test)
if [ ! -f harness_split/outputs/preparation.json ]; then
  log "5. 모델·데이터 받기"
  "$PY" harness_split/src/prepare.py --existing-env --config harness_split/src/config_test_seed3407.json
fi

log "준비 끝. 평가: nohup bash run_test.sh <DPO2E_repro_adapter_eval.zip> <seed> > test_<seed>.log 2>&1 &"
