"""DPO2-E 어댑터를 베이스에 병합하고 DeepStack 게이트를 흡수한다 (E의 train_dpo_d.py merge_local과 같은 방식).

사용: python merge_dpo2e.py <adapter 폴더> <E manifest.json> <출력 폴더>
끝나면 E가 MMMU-val 평가에 쓴 병합 모델과 SHA-256을 비교해 <출력 폴더>/.merge_ok에 적는다.
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

BASE_REPO, BASE_REV = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"
BASE_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
              "preprocessor_config.json", "video_preprocessor_config.json"]

adapter, manifest, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cache = os.environ.get("MMMU_MODEL_CACHE") or str(Path("~/.cache/huggingface/hub").expanduser())  # 하네스와 같은 캐시
base = Path(snapshot_download(BASE_REPO, revision=BASE_REV, cache_dir=cache, local_files_only=True))
gates = json.loads((adapter / "gates.json").read_text())["deepstack_gates"]

model = Qwen3VLForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16)
merged = PeftModel.from_pretrained(model, adapter).merge_and_unload()
with torch.no_grad():  # 게이트를 deepstack merger의 linear_fc2 가중치·편향에 곱한다
    for l, g in enumerate(gates):
        fc2 = merged.model.visual.deepstack_merger_list[l].linear_fc2
        fc2.weight.mul_(g)
        fc2.bias.mul_(g)
if out.exists():
    shutil.rmtree(out)
merged.save_pretrained(out, safe_serialization=True)
for n in BASE_FILES:
    if (base / n).exists():
        shutil.copy2(base / n, out / n)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


ref = json.loads(manifest.read_text(encoding="utf-8"))["model_identity"]
rows = [("config.json", sha(out / "config.json"), ref["config_sha256"])]
rows += [(n, sha(out / n) if (out / n).exists() else None, v) for n, v in ref["weights_sha256"].items()]
for n, a, b in rows:
    print(f"  {n}: {'같음' if a == b else '다름'}")
same = all(a == b for _, a, b in rows) and \
    sorted(p.name for p in out.glob("*.safetensors")) == sorted(ref["weights_sha256"])
msg = ("병합 모델이 E가 val 평가에 쓴 모델과 SHA-256까지 같다" if same else
       "경고: 병합 모델이 E가 val 평가에 쓴 모델과 SHA-256이 다르다 (패키지 버전 차이 등). 평가는 계속한다")
print(msg)
(out / ".merge_ok").write_text(msg + "\n", encoding="utf-8")
