"""끊길 때 jsonl 마지막 줄이 반쯤 쓰였으면 그 줄을 따로 빼 둔다 (하네스는 깨진 줄이 있으면 이어 하지 못한다)."""
import json
import sys
import time

p = sys.argv[1]
lines = open(p, encoding="utf-8").read().splitlines(keepends=True)
if lines:
    try:
        json.loads(lines[-1])
    except ValueError:
        open(p + f".partial_{int(time.time())}", "w", encoding="utf-8").write(lines[-1])
        open(p, "w", encoding="utf-8").write("".join(lines[:-1]))
        print(f"반쯤 쓰인 마지막 줄을 빼 둠: {p}")
