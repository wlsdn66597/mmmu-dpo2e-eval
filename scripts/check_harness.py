"""평가 하네스 코드가 E의 MMMU-val 평가 기록(manifest.json code_sha256)과 같은지 확인한다."""
import hashlib
import sys
from pathlib import Path

EXPECTED = {  # DPO2E_repro_adapter_eval.zip의 eval_mmmu_val/manifest.json에서 옮김
    "src/config.json": "b55e958623cd59e3da64aef4e228a98e83e586d924b80a8b3fd22dafd7c938e5",
    "src/evaluate.py": "2dbc17e8953cbdf1ea1091e2890489c7f0592efe21e5d56a13bf1b61b642e75d",
    "src/judge.py": "443a28c9cbfdd90fa6f6c5021a557dc97d58df3bed7967318ec972b773e3c0fb",
    "src/parsers.py": "5023ed38d2e8af91304b258d5b3798ffc03df24324f49740953d943378a30b04",
    "src/prepare.py": "581b4c2905fbd71f77c3e5f6f23a13c4444a125609a3c9b3ac4bfbe267f476ef",
}
root = Path(sys.argv[1])
bad = [k for k, v in EXPECTED.items() if hashlib.sha256((root / k).read_bytes()).hexdigest() != v]
if bad:
    sys.exit(f"하네스 코드가 E 평가 때와 다르다: {bad}")
print("하네스 코드 5개가 E 평가 기록의 해시와 같다")
