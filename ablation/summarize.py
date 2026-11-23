"""val 결과(harness results.json)들을 ablation 표로 모은다.

사용: python summarize.py <결과 폴더(results.json 있는 곳)>... > ablation_summary.md
"""
import json, re, statistics, sys
from pathlib import Path

print("| run | 정답 /900 | 상한 도달 | 길이 중앙값 | parsed-answer acc. | 답만 낸 응답 |")
print("|---|---|---|---|---|---|")
for d in sys.argv[1:]:
    run = Path(d).name
    p = Path(d) / "results.json"
    if not p.is_file():
        print(f"| {run} | (결과 없음) | | | | |")
        continue
    recs = json.loads(p.read_text(encoding="utf-8"))["records"]
    hit = sum(bool(r["grading"]["final"]["hit"]) for r in recs)
    length = sum(r["model_response"]["finish_reason"] == "length" for r in recs)
    med = statistics.median(r["model_response"]["output_tokens"] for r in recs)
    parsed = [r for r in recs if r["grading"]["final"]["method"].startswith("explicit_final")]
    pacc = sum(bool(r["grading"]["final"]["hit"]) for r in parsed)
    only = sum(bool(re.fullmatch(r"\s*Final answer:\s*\S+\s*", r["model_response"]["raw"])) for r in recs)
    print(f"| {run} | {hit} | {length} | {med} | {100 * pacc / max(1, len(parsed)):.2f}% ({pacc}/{len(parsed)}) | {only} |")
