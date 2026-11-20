# DPO2-E 모델 MMMU test 평가 (E와 같은 하네스·설정, seed만 변경)

새 Linux GPU 컴퓨터에서 이 저장소와 `DPO2E_repro_adapter_eval.zip`만으로 환경 설정, 모델·데이터 다운로드, 병합, MMMU test 평가까지 진행한다.
모델 가중치, 문항, 이미지는 저장소에 없다. 모두 고정 revision으로 내려받는다.

## 실행

```bash
git clone <이 저장소 주소> mmmu-dpo2e-eval
cd mmmu-dpo2e-eval
bash setup.sh                                   # 환경 + 하네스 + 모델·데이터 (한 번만)
nohup bash run_test.sh ~/DPO2E_repro_adapter_eval.zip 1234 > test_1234.log 2>&1 &
```

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
| 기타 | git, 인터넷, 디스크 약 60GB |
| 시간 | E의 val 900문항이 약 5.5시간이었다. test 10,500문항은 60시간 이상 예상 |

## 구성

| 경로 | 내용 |
|---|---|
| `setup.sh` | 가상환경(`.venv`), 패키지, 하네스(`harness/`), test용 복사본(`harness_split/`), 모델·데이터 다운로드 |
| `run_test.sh` | zip 풀기(`dpo2e/`), 병합(`dpo2e/merged/`), 평가·채점 |
| `env/requirements.txt` | 하네스의 고정 패키지 목록 (vLLM 0.11.0, torch 2.8.0, transformers 4.57.0 등 157개) |
| `env/requirements_merge.txt` | 병합용 peft 0.17.1, accelerate 1.10.1 (E 학습 환경과 같은 버전) |
| `scripts/check_harness.py` | 하네스 코드 5개가 E 평가 기록의 SHA-256과 같은지 확인 |
| `scripts/patch_harness.py` | 하네스를 test에서 돌 수 있게 고침 |
| `scripts/make_config.py` | test용 config 만들기 |
| `scripts/merge_dpo2e.py` | 어댑터 병합 + 게이트 흡수, E 병합 모델과 SHA-256 비교 |
| `scripts/trim_partial.py` | 끊겨서 반쯤 쓰인 마지막 줄 정리 |

## E 평가와 같은 것, 바꾼 것

같은 것
- 평가 하네스: `wlsdn66597/mmdl-baseline` 커밋 `7740786` (E manifest의 git_commit, 코드 해시 확인)
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
