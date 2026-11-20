"""평가 하네스(mmdl-baseline @7740786)를 MMMU test에서 돌 수 있게 고친다. 원본은 val 900문항 전용이다.

사용: python patch_harness.py <하네스 복사본 폴더>
생성·판정·채점 로직, 프롬프트, 샘플링은 건드리지 않는다. 바꾼 곳은 <폴더>/PATCH_NOTES.md에 적는다.
"""
import re, sys
from pathlib import Path
root = Path(sys.argv[1])

def sub(path, old, new, regex=False):
    p = root / path
    s = p.read_text(encoding="utf-8")
    n = len(re.findall(old, s, flags=re.S)) if regex else s.count(old)
    if n != 1:
        sys.exit(f"{path}: 고칠 곳을 정확히 1곳 찾지 못함 ({n}곳): {old[:60]!r}")
    s = re.sub(old, lambda _: new, s, flags=re.S) if regex else s.replace(old, new)
    p.write_text(s, encoding="utf-8")

# prepare.py: split을 config에서 읽고, test는 10,500문항(과목별 수는 제각각)으로 확인
sub("src/prepare.py", "def config(path=None):",
    "def expected_total(c):\n    return {'validation': 900, 'test': 10500}[c['dataset']['split']]\n\n\ndef config(path=None):")
sub("src/prepare.py", "assert c['dataset']['split'] == 'validation' and",
    "assert c['dataset']['split'] in ('validation', 'test') and")
sub("src/prepare.py",
    "subject, split='validation',\n            revision=c['dataset']['revision'], cache_dir=cache)\n        if len(ds) != 30 or not ds.cache_files",
    "subject, split=c['dataset']['split'],\n            revision=c['dataset']['revision'], cache_dir=cache)\n        if (c['dataset']['split'] == 'validation' and len(ds) != 30) or not ds.cache_files")
sub("src/prepare.py", "row['id'].startswith('validation_' + subject + '_')",
    "row['id'].startswith(c['dataset']['split'] + '_' + subject + '_')")
sub("src/prepare.py", "    if len(rows) != 900:\n        raise RuntimeError('Expected 900 questions.')",
    "    if len(rows) != expected_total(c):\n        raise RuntimeError(f'Expected {expected_total(c)} questions.')")

# evaluate.py: 문항 ID·개수 확인, 진행 표시, 점수 집계(전체 문항 기준 + 과목 평균 같이)
sub("src/evaluate.py", "r['id'].startswith('validation_' + r['subject'] + '_')",
    "r['id'].startswith(c['dataset']['split'] + '_' + r['subject'] + '_')")
sub("src/evaluate.py",
    "    if any(n>30 for n in counts.values()) or (not partial and (len(records)!=900 or any(counts[s]!=30 for s in c['dataset']['configs']))):\n        raise RuntimeError('Expected every one of 30 subjects with 30 questions.')",
    "    if c['dataset']['split'] == 'validation':\n"
    "        if any(n>30 for n in counts.values()) or (not partial and (len(records)!=900 or any(counts[s]!=30 for s in c['dataset']['configs']))):\n"
    "            raise RuntimeError('Expected every one of 30 subjects with 30 questions.')\n"
    "    elif not partial and (len(records)!=prepare.expected_total(c) or any(counts[s]==0 for s in c['dataset']['configs'])):\n"
    "        raise RuntimeError(f'Expected all {prepare.expected_total(c)} questions of the 30 subjects.')")
sub("src/evaluate.py", "print('Inference complete; reusing all 900 saved responses.',flush=True)",
    "print(f'Inference complete; reusing all {len(rows)} saved responses.',flush=True)")
sub("src/evaluate.py", "print(f'Inference {len(done)+1}/900 ", "print(f'Inference {len(done)+1}/{len(rows)} ")
sub("src/evaluate.py", r"def summaries\(records,c\):.*?'exact_official_report_reproduction':False\}\n", '''def summaries(records,c):
    check_records(records,c)
    if any(r['grading']['final']['status'] != 'graded' for r in records):
        raise RuntimeError('Incomplete grading; no final accuracy.')
    total = len(records)
    correct = sum(r['grading']['final']['hit'] for r in records)
    subjects = {}
    for subject in c['dataset']['configs']:
        items = [r for r in records if r['subject']==subject]
        n = sum(r['grading']['final']['hit'] for r in items)
        subjects[subject] = {'total':len(items),'correct':n,'incorrect':len(items)-n,'accuracy':n/len(items)}
    macro = sum(v['accuracy'] for v in subjects.values())/len(subjects)
    if c['dataset']['split']=='validation' and not math.isclose(macro,correct/total,abs_tol=1e-12):
        raise RuntimeError('Macro/micro arithmetic mismatch.')
    return {'split':c['dataset']['split'],'total':total,'graded':total,'correct':correct,'incorrect':total-correct,
        'accuracy':correct/total,'accuracy_percent':100*correct/total,
        'aggregation':'all questions pooled (equals the subject mean on validation)',
        'subject_mean_accuracy_percent':100*macro,
        'by_subject':subjects,'method_counts':dict(Counter(r['grading']['final']['method'] for r in records)),
        'length_limit_incorrect':sum(r['model_response']['finish_reason']=='length' for r in records),
        'judge_count':sum(r['grading']['final']['method']=='local_qwen3_8b_judge_parser' for r in records),
        'judge_states':dict(Counter(r['grading']['final'].get('judge_parser_decision',{}).get('status') for r in records if 'judge_parser_decision' in r['grading']['final'])),
        'random_fallback':0,'exact_official_report_reproduction':False}
''', regex=True)
sub("src/evaluate.py",
    "    table += [f'| {s} | 30 | {v[\"correct\"]} | {v[\"accuracy\"]*100:.2f}% |' for s,v in summary['by_subject'].items()]\n"
    "    table += [f'| Overall (macro avg) | 900 | {summary[\"correct\"]} | {summary[\"accuracy_percent\"]:.2f}% |']",
    "    table += [f'| {s} | {v[\"total\"]} | {v[\"correct\"]} | {v[\"accuracy\"]*100:.2f}% |' for s,v in summary['by_subject'].items()]\n"
    "    table += [f'| Overall (all questions) | {summary[\"total\"]} | {summary[\"correct\"]} | {summary[\"accuracy_percent\"]:.2f}% |',\n"
    "              f'| Subject mean | {summary[\"total\"]} | - | {summary[\"subject_mean_accuracy_percent\"]:.2f}% |']")

(root / "PATCH_NOTES.md").write_text("""# test용으로 바꾼 곳 (원본: mmdl-baseline @7740786)

- prepare.py: 데이터 split을 config의 dataset.split에서 읽는다. test는 문항 ID가 `test_`로 시작하고 전체 10,500문항인지 확인한다
  (val은 원래대로 과목당 30문항·900문항 확인).
- evaluate.py: 문항 확인을 split에 맞게, 진행 표시 분모를 실제 문항 수로 바꿨다.
  점수는 전체 문항 기준 정답률(accuracy)과 과목 평균(subject_mean_accuracy_percent)을 같이 낸다. val에서는 둘이 같다.
- config_test_seed<seed>.json: 원본 config.json에서 dataset.split=test, sampling.seed=<seed>,
  runtime.max_images_per_prompt=7 (test에 이미지 6·7장 문항이 10개 있다)만 바꿨다.
- 생성·판정·채점 코드, 프롬프트, 샘플링(seed 외), 이미지 크기, 출력 상한, 1문항씩 생성은 그대로다.
""", encoding="utf-8")
print("하네스 test용 수정 완료")
