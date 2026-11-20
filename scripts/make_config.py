"""하네스 config.json에서 test용 설정을 만든다. 바꾸는 것은 split, 생성 seed, 이미지 상한뿐이다.

사용: python make_config.py <원본 config.json> <출력 경로> <seed>
"""
import json
import sys

src, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(src, encoding="utf-8"))
c["dataset"]["split"] = "test"
c["sampling"]["seed"] = seed
c["runtime"]["max_images_per_prompt"] = 7  # test에 이미지 6·7장 문항이 10개 있다 (val은 최대 5장)
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(json.dumps(c, ensure_ascii=False, indent=2) + "\n")
