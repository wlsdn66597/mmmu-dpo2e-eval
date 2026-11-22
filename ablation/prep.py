"""Ablation 준비: E 학습 코드를 작업 폴더로 복사·패치하고, 입력 데이터를 검사하고, 변형별 쌍 파일을 만든다.

사용: python prep.py <repo 루트> <pairs_E.jsonl> <이미지 루트> <작업 폴더>
- E zip을 푼 <repo>/dpo2e/code/{train_dpo_d.py, common.py}, <repo>/dpo2e/logs/train_info.json 을 읽는다.
- train_dpo_d.py 패치 두 가지 (E의 동작은 바꾸지 않는다):
  1) ABL_SCORE_FROM_START=1 이면 loop 쌍의 rejected 점수를 응답 처음부터 계산 (기본은 E와 같이 loop onset부터)
  2) ABL_ATTN 으로 attention 구현 지정 (기본 flash_attention_2, flash_attn이 없으면 sdpa)
- train_sft.py: train_dpo_d.py가 가져다 쓰는 LORA_TARGET, BASE_FILES, md5 만 담은 최소 모듈
- training_settings_only.json: E의 train_info.json 'vision' 설정 그대로
"""
import hashlib, json, os, shutil, sys
from pathlib import Path

repo, pairs, images, work = (Path(a) for a in sys.argv[1:5])
code_src = repo / "dpo2e" / "code"
info = json.loads((repo / "dpo2e" / "logs" / "train_info.json").read_text(encoding="utf-8"))
code = work / "code"
code.mkdir(parents=True, exist_ok=True)


def fail(msg):
    sys.exit(f"[prep] 중단: {msg}")


# ---- 1. 입력 검사
if not pairs.is_file():
    fail(f"쌍 파일이 없다: {pairs}")
h = hashlib.md5(pairs.read_bytes()).hexdigest()
rows = [json.loads(l) for l in pairs.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"[prep] 쌍 파일 {len(rows)}행, md5 {h} (E 재현 학습: {info['n_pairs']}행, md5 {info['pairs_md5']})")
if h != info["pairs_md5"] or len(rows) != info["n_pairs"]:
    if "--force" not in sys.argv:
        fail("E가 학습한 쌍 파일과 다르다. 그래도 진행하려면 FORCE=1 로 실행")
    print("[prep] 경고: E와 다른 쌍 파일로 진행 (FORCE=1)")
# loop 쌍은 E 학습 서버의 절대 경로(/shared_dir/friend_dpo_r4/x/...)를 가리킨다. 팀원 묶음(pairs_E_bundle.tar)은
# 같은 구조를 쌍 파일 옆 mmdl/ 에 두었으므로 그쪽으로 바꾼다. 다른 위치면 IMG_REMAP="옛접두어=새접두어"
old, new = (os.environ.get("IMG_REMAP") or f"/shared_dir/friend_dpo_r4/x/={pairs.parent / 'mmdl'}/").split("=", 1)
n_remap = 0
for r in rows:
    if r["image"].startswith(old):
        r["image"] = new + r["image"][len(old):]; n_remap += 1
print(f"[prep] 이미지 절대 경로 {n_remap}행: {old} → {new}")
missing = [r["image"] for r in rows if not (images / r["image"]).is_file()]
if missing:
    fail(f"이미지 {len(missing)}개가 없다 (상대 경로는 {images} 기준). 예: {missing[:3]}")
loop_rows = [r for r in rows if r.get("kind") != "replay" and r.get("rej_span_char") is not None]
print(f"[prep] loop 쌍(rej_span 있음) {len(loop_rows)}행, replay {sum(r.get('kind') == 'replay' for r in rows)}행, "
      f"나머지(ERA 쌍) {len(rows) - len(loop_rows) - sum(r.get('kind') == 'replay' for r in rows)}행")
if len(loop_rows) != 1304 and "--force" not in sys.argv:
    fail(f"loop 쌍이 1,304행이어야 하는데 {len(loop_rows)}행이다")

# ---- 2. 학습용 쌍 파일 (이미지 경로만 바꾼 전체, loop 쌍만 뽑은 mainonly용)
(work / "pairs_full.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
(work / "pairs_loop_only.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in loop_rows),
                                            encoding="utf-8")

# ---- 3. 학습 코드 복사·패치
src = (code_src / "train_dpo_d.py").read_text(encoding="utf-8")
patches = [
    ("import argparse, hashlib, json, math, random, shutil, sys, time",
     "import argparse, hashlib, json, math, os, random, shutil, sys, time"),
    ("                j = j[:, :hi]\n",
     "                j = j[:, :hi]\n"
     "                if os.environ.get(\"ABL_SCORE_FROM_START\") == \"1\":  # ablation: 루프 앞 풀이까지 점수에 포함\n"
     "                    lo = 0\n"),
    ('attn_implementation="flash_attention_2"',
     'attn_implementation=os.environ.get("ABL_ATTN", "flash_attention_2")'),
]
for old, new in patches:
    if src.count(old) != 1:
        fail(f"train_dpo_d.py 패치 위치를 정확히 1곳 찾지 못함: {old[:50]!r}")
    src = src.replace(old, new)
(code / "train_dpo_d.py").write_text(src, encoding="utf-8")
shutil.copy2(code_src / "common.py", code / "common.py")
(code / "train_sft.py").write_text('''import hashlib
LORA_TARGET = r".*language_model.*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
BASE_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
              "preprocessor_config.json", "video_preprocessor_config.json"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
''', encoding="utf-8")
(code / "training_settings_only.json").write_text(
    json.dumps({"evaluation": {"settings": {"vision": info["vision"]}}, "records": []}, ensure_ascii=False, indent=1),
    encoding="utf-8")
print(f"[prep] 준비 완료: {work}")
