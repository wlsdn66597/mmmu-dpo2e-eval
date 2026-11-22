# DPO2-E 모델 MMMU test 평가 (E와 같은 하네스·설정, seed만 변경)

새 Linux GPU 컴퓨터에서 이 저장소와 `DPO2E_repro_adapter_eval.zip`만으로 환경 설정, 모델·데이터 다운로드, 병합, MMMU test 평가까지 진행한다.
모델 가중치, 문항, 이미지는 저장소에 없다. 모두 고정 revision으로 내려받는다.

## 실행

```bash
git clone https://github.com/wlsdn66597/mmmu-dpo2e-eval.git
cd mmmu-dpo2e-eval
bash setup.sh                                   # 환경 + 하네스 + 모델·데이터 (한 번만)
nohup bash run_test.sh ~/DPO2E_repro_adapter_eval.zip 1234 > test_1234.log 2>&1 &
```

- MMMU val(900문항)은 앞에 `SPLIT=validation`을 붙인다. seed 3407이면 E의 val 평가 설정과 완전히 같다 (E 재현 모델 602/900).
  `SPLIT=validation nohup bash run_test.sh ~/DPO2E_repro_adapter_eval.zip 3407 > val_3407.log 2>&1 &`
  val 데이터가 캐시에 없으면 먼저 받는다. 결과는 `harness_split/outputs/dpo2E_val_seed<seed>/`
- 둘 다 끊기면 같은 명령을 다시 실행하면 끝난 단계는 건너뛰고 이어서 한다.
- 결과: `harness_split/outputs/dpo2E_test_seed<seed>/scores.md`(과목별·전체 점수), `summary.json`, `predictions.jsonl`(문항별 응답), `results.json`(채점 포함)
- seed를 바꿔 여러 번 돌릴 때는 `run_test.sh`만 다시 실행한다. 병합 모델과 다운로드는 다시 하지 않는다.
- 모델·데이터 캐시는 `~/.cache/huggingface`다. 다른 디스크에 두려면 두 스크립트 모두 실행 전에 `export MMMU_MODEL_CACHE=/경로/hub MMMU_DATASET_CACHE=/경로/datasets`를 한다.

## 필요한 것

| 항목 | 값 |
|---|---|
| OS | Linux x86-64 (하네스가 확인한다) |
| GPU | 24GB 1장, bf16 지원 (E는 RTX 4090). 평가 중 다른 프로세스가 GPU를 쓰지 않을 것 |
| Python | 3.12 (`python3.12`가 없으면 `uv`로 만든다) |
| 기타 | git, 인터넷, 디스크 약 60GB. GitHub 로그인은 필요 없다 |
| 시간 | E의 val 900문항이 약 5.5시간이었다. test 10,500문항은 60시간 이상 예상 |

## 구성

| 경로 | 내용 |
|---|---|
| `setup.sh` | 가상환경(`.venv`), 패키지, 하네스(`harness/`), test용 복사본(`harness_split/`), 모델·데이터 다운로드 |
| `run_test.sh` | zip 풀기(`dpo2e/`), 병합(`dpo2e/merged/`), 평가·채점 |
| `harness_src/` | E가 MMMU-val을 평가한 하네스 코드 (mmdl-baseline 커밋 `7740786`의 `src/` 5개 파일과 라이선스 고지, 수정 없음) |
| `env/requirements.txt` | 하네스의 고정 패키지 목록 (vLLM 0.11.0, torch 2.8.0, transformers 4.57.0 등 157개) |
| `env/requirements_merge.txt` | 병합용 peft 0.17.1, accelerate 1.10.1 (E 학습 환경과 같은 버전) |
| `scripts/check_harness.py` | 하네스 코드 5개가 E 평가 기록의 SHA-256과 같은지 확인 |
| `scripts/patch_harness.py` | 하네스를 test에서 돌 수 있게 고침 |
| `scripts/make_config.py` | test용 config 만들기 |
| `scripts/merge_dpo2e.py` | 어댑터 병합 + 게이트 흡수, E 병합 모델과 SHA-256 비교 |
| `scripts/trim_partial.py` | 끊겨서 반쯤 쓰인 마지막 줄 정리 |

## E 평가와 같은 것, 바꾼 것

같은 것
- 평가 하네스: mmdl-baseline 커밋 `7740786` (E manifest의 git_commit). `harness_src/`에 넣어 두었고, setup.sh가 코드 5개의 SHA-256이 E 기록과 같은지 확인한다
- 베이스 `Qwen/Qwen3-VL-4B-Instruct@ebb281ec`, 데이터 `MMMU/MMMU@98e6ac0c`, 판정 `Qwen/Qwen3-8B@b968826d`
- 프롬프트, temperature 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5, 출력 상한 32,768, 문맥 64,000
- 이미지 min_pixels 1,003,520 / max_pixels 4,014,080, 1문항씩 생성, 판정 최대 25번, 채점 규칙
- 병합 방식: peft `merge_and_unload` 후 게이트 값을 deepstack merger `linear_fc2`의 가중치·편향에 곱함

바꾼 것 (자세한 내용은 실행 후 `harness_split/PATCH_NOTES.md`)
- 데이터 분할 validation → test, 문항 수 확인 900 → 10,500
- 생성 seed 3407 → 지정한 값 (판정 모델 seed는 그대로)
- 이미지 상한 5 → 7: test에 이미지 6·7장 문항이 10개 있다. 나머지 문항에는 영향이 없다
- 점수: 전체 문항 기준 정답률(accuracy)과 과목 평균을 같이 낸다. val은 과목당 30문항이라 둘이 같았지만 test는 과목별 문항 수가 달라 다르다

## 확인한 것

- 고친 하네스로 E의 val 기록을 다시 집계하면 원래 점수 602/900이 그대로 나온다.
- 가짜 test 10,500문항으로 집계와 누락 검출을 시험했다.
- 새 컴퓨터에서 설치·다운로드·병합·GPU 평가까지 처음부터 돌려 보지는 않았다.
- 병합 모델이 E와 SHA-256까지 같은지는 `run_test.sh` 2단계에서 출력된다.

## 알아 둘 것

- 판정 입력이 판정 모델 문맥(32,768)을 넘으면 하네스가 멈춘다 (E 하네스 그대로). 답 줄 없이 아주 길게 끝난 응답에서만 생긴다.
- 같은 seed라도 GPU·드라이버가 다르면 응답이 일부 달라질 수 있다.

## Ablation (E recipe, MMMU val seed 3407)

`ablation/run_ablations.sh`가 변형 3개를 차례로 학습 → 병합 → val 평가한다. E 학습 코드는 E zip의 `code/`를 복사해 작업 폴더에서만 고친다 (`ablation/prep.py` 참고). 다음 두 가지가 필요하다.

1. setup.sh, run_test.sh를 한 번 돌려 `dpo2e/`가 풀려 있는 상태
2. E가 학습한 쌍 파일 `pairs_E.jsonl` (2,307행, md5 `9b7b6d94abae89ee5862ced67d3c69c2`)과 그 쌍의 `image` 경로가 가리키는 이미지 폴더. 팀원 묶음 `pairs_E_bundle.tar`를 `~/pairsE`에 풀면 `~/pairsE/pairs_E.jsonl ~/pairsE/vic_era_2000`을 주면 된다 (loop 쌍의 절대 경로는 묶음의 `mmdl/`로 자동으로 바꾼다)

| 이름 | E와 다른 점 |
|---|---|
| prefix | loop 쌍의 rejected 점수를 loop onset이 아니라 응답 처음부터 계산 (입력 토큰은 같음) |
| mainonly | loop 쌍 ×4 + chosen NLL만 학습 (ERA 쌍·replay·시각 연결부 LoRA·gate·이미지 조건 손실 제거) |
| novision | 시각 연결부 LoRA·gate·이미지 조건 손실만 제거 |

```bash
CC=$HOME/bin/cc setsid nohup bash ablation/run_ablations.sh <pairs_E.jsonl> <이미지 루트> > ablation.log 2>&1 < /dev/null &
```

GPU에 다른 평가가 돌고 있으면 끝날 때까지 5분마다 기다렸다가 시작한다. 끊기면 같은 명령으로 다시 실행하면 이어서 한다. 결과 표는 `ablation_work/ablation_summary.md`(E 재현 val 결과와 함께).
