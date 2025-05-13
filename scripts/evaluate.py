#!/usr/bin/env python
import sys
from pathlib import Path

# 1) Projekt-Root ins PYTHONPATH einfügen
root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(root))

# 2) Nun kann das app-Package importiert werden
from app.rag_chain import answer_query

import json

# 3) Test-Set laden
test_file = root / "data" / "test_questions.json"
if not test_file.exists():
    raise FileNotFoundError(f"Test file not found: {test_file}")

with open(test_file, encoding="utf-8") as f:
    tests = json.load(f)

# 4) Evaluation durchführen
results = []
for t in tests:
    pred = answer_query(t["question"])
    results.append({
        "question": t["question"],
        "expected": t["answer"],
        "predicted": pred
    })

# 5) Ergebnisse speichern
out_file = root / "data" / "eval_results.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"✅ Evaluation completed, results saved to {out_file}")