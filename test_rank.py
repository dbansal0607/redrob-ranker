import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import json
"""
Unit tests for rank.py
Run: python test_rank.py
"""

import csv
import tempfile
import os

# ── import the module under test ──────────────────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location("rank", "rank.py")
rank = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rank)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def check(name, condition):
    status = PASS if condition else FAIL
    results.append((name, status))
    print(f"  {status}  {name}")


# ── Helpers ───────────────────────────────────────────────────────

def make_candidate(
    cid="CAND_TEST_001",
    title="Senior ML Engineer",
    company="ProductCo",
    company_size="startup",
    yoe=7.0,
    location="Pune",
    country="India",
    skills=None,
    career=None,
    signals=None,
    summary=""
):
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "current_company": company,
            "current_company_size": company_size,
            "years_of_experience": yoe,
            "location": location,
            "country": country,
            "summary": summary,
            "headline": "",
        },
        "skills": skills or [
            {"name": "Python", "proficiency": "expert", "duration_months": 60},
            {"name": "FAISS", "proficiency": "expert", "duration_months": 36},
            {"name": "sentence-transformers", "proficiency": "advanced", "duration_months": 24},
            {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 30},
        ],
        "career_history": career or [
            {
                "title": "ML Engineer",
                "company": "ProductCo",
                "duration_months": 36,
                "description": "Built embedding-based retrieval system, shipped to production, ran A/B tests, measured NDCG."
            },
            {
                "title": "Data Scientist",
                "company": "StartupX",
                "duration_months": 48,
                "description": "Developed ranking pipeline using vector search and reranking."
            }
        ],
        "certifications": [],
        "redrob_signals": signals or {
            "open_to_work_flag": True,
            "last_active_date": "2026-06-15",
            "recruiter_response_rate": 0.85,
            "notice_period_days": 30,
            "interview_completion_rate": 0.9,
            "github_activity_score": 70,
            "profile_completeness_score": 85,
            "willing_to_relocate": False,
        }
    }


# ════════════════════════════════════════════════════════════════
print("\n── Test Suite: rank.py ─────────────────────────────────\n")

# ── 1. Honeypot Detection ────────────────────────────────────────
print("1. Honeypot Detection")

hp1 = make_candidate()
hp1["skills"] = [
    {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 0}
    for i in range(6)
]
check("≥5 expert skills with 0 months → honeypot", rank.is_honeypot(hp1))

hp2 = make_candidate()
hp2["skills"] = [
    {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 1}
    for i in range(4)
]
check("4 expert skills with 1 month → honeypot", rank.is_honeypot(hp2))

hp3 = make_candidate(yoe=10.0)
hp3["career_history"] = [
    {"title": "Eng", "company": "Co", "duration_months": 12, "description": ""}
]
check("Career months << claimed YOE → honeypot", rank.is_honeypot(hp3))

# Rule 4 removed — 10+ expert skills is normal for senior 8+ yr engineers
# (was causing false positives disqualifying real top talent)
# hp4 test intentionally removed per review feedback

normal = make_candidate()
check("Normal strong candidate → NOT honeypot", not rank.is_honeypot(normal))

# ── 2. Experience Band Scoring ───────────────────────────────────
print("\n2. Experience Band Scoring")

check("YOE=7 → score 1.0",   rank.score_experience_band(make_candidate(yoe=7.0)) == 1.0)
check("YOE=6 → score 1.0",   rank.score_experience_band(make_candidate(yoe=6.0)) == 1.0)
check("YOE=5 → score 0.88",  rank.score_experience_band(make_candidate(yoe=5.0)) == 0.88)
check("YOE=9 → score 0.88",  rank.score_experience_band(make_candidate(yoe=9.0)) == 0.88)
check("YOE=2 → score < 0.5", rank.score_experience_band(make_candidate(yoe=2.0)) < 0.5)
check("YOE=15 → score < 0.5",rank.score_experience_band(make_candidate(yoe=15.0)) < 0.5)

# ── 3. Career Trajectory — Wrong Title Hard Gate ─────────────────
print("\n3. Career Trajectory — Wrong Title Gate")

hr = make_candidate(title="HR Manager")
hr_score = rank.score_career_trajectory(hr, rank.build_candidate_text(hr))
check("HR Manager title → career score ≈ 0.04", abs(hr_score - 0.04) < 0.01)

mkt = make_candidate(title="Marketing Manager")
mkt_score = rank.score_career_trajectory(mkt, rank.build_candidate_text(mkt))
check("Marketing Manager → career score ≈ 0.04", abs(mkt_score - 0.04) < 0.01)

ml_eng = make_candidate(title="ML Engineer")
ml_score = rank.score_career_trajectory(ml_eng, rank.build_candidate_text(ml_eng))
check("ML Engineer → career score > 0.5", ml_score > 0.5)

# ── 4. Consulting Penalty ────────────────────────────────────────
print("\n4. Consulting Firm Penalty")

tcs_candidate = make_candidate()
tcs_candidate["career_history"] = [
    {"title": "ML Engineer", "company": "TCS", "duration_months": 60, "description": "ML work"},
    {"title": "Data Scientist", "company": "Infosys", "duration_months": 36, "description": "DS work"},
]
tcs_score = rank.score_career_trajectory(tcs_candidate, rank.build_candidate_text(tcs_candidate))
check("100% consulting career → career score < 0.45", tcs_score < 0.45)

mixed = make_candidate()
mixed["career_history"] = [
    {"title": "ML Engineer", "company": "ProductStartup", "duration_months": 48, "description": "Built retrieval system"},
    {"title": "Analyst", "company": "TCS", "duration_months": 12, "description": "Analytics"},
]
mixed_score = rank.score_career_trajectory(mixed, rank.build_candidate_text(mixed))
check("Mostly product + short consulting stint → career score > 0.5", mixed_score > 0.5)

# ── 5. Location Scoring ──────────────────────────────────────────
print("\n5. Location Scoring")

pune = make_candidate(location="Pune", country="India")
check("Pune, India → location score 1.0", rank.score_location_availability(pune) == 1.0)

noida = make_candidate(location="Noida", country="India")
check("Noida, India → location score 1.0", rank.score_location_availability(noida) == 1.0)

hyd = make_candidate(location="Hyderabad", country="India")
check("Hyderabad, India → location score 0.9", rank.score_location_availability(hyd) == 0.90)

abroad = make_candidate(location="London", country="UK")
abroad["redrob_signals"]["willing_to_relocate"] = False
check("Non-India, not relocating → score 0.15", rank.score_location_availability(abroad) == 0.15)

# ── 6. Behavioral Signals ────────────────────────────────────────
print("\n6. Behavioral Signals")

active = make_candidate()
active["redrob_signals"]["last_active_date"] = "2026-06-17"
active["redrob_signals"]["open_to_work_flag"] = True
active["redrob_signals"]["recruiter_response_rate"] = 1.0
active["redrob_signals"]["notice_period_days"] = 0
active_score = rank.score_behavioral_signals(active)
check("Perfect behavioral signals → score > 0.85", active_score > 0.85)

ghost = make_candidate()
ghost["redrob_signals"]["last_active_date"] = "2024-01-01"
ghost["redrob_signals"]["open_to_work_flag"] = False
ghost["redrob_signals"]["recruiter_response_rate"] = 0.0
ghost["redrob_signals"]["notice_period_days"] = 180
ghost_score = rank.score_behavioral_signals(ghost)
check("Ghost candidate → behavioral score < 0.30", ghost_score < 0.30)

# ── 7. Full Pipeline — compute_score ────────────────────────────
print("\n7. Full Pipeline")

strong = make_candidate()
score, bd = rank.compute_score(strong)
check("Strong candidate → total score > 0.7", score > 0.7)
check("Score keys present", all(k in bd for k in ["skills", "career", "experience", "behavioral", "location", "total"]))

honeypot_c = make_candidate()
honeypot_c["skills"] = [
    {"name": f"S{i}", "proficiency": "expert", "duration_months": 0} for i in range(6)
]
hp_score, hp_bd = rank.compute_score(honeypot_c)
check("Honeypot → composite score ≈ 0.001", hp_score <= 0.002)
check("Honeypot → flagged in breakdown", hp_bd.get("honeypot") is True)

# ── 8. Ranking Monotonicity (end-to-end mini run) ────────────────
print("\n8. End-to-End Mini Run")

candidates = [
    make_candidate(cid=f"CAND_{i:07d}", yoe=float(5 + i % 5))
    for i in range(20)
]
# Add one honeypot
hp = make_candidate(cid="CAND_HP_001")
hp["skills"] = [{"name": f"S{i}", "proficiency": "expert", "duration_months": 0} for i in range(6)]
candidates.append(hp)

with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
    for c in candidates:
        f.write(json.dumps(c) + "\n")
    tmp_in = f.name

tmp_out = tmp_in.replace(".jsonl", "_out.csv")

# Run
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
sys.argv = ["rank.py", "--candidates", tmp_in, "--out", tmp_out, "--topn", "10"]
try:
    rank.main()
    with open(tmp_out) as f:
        rows = list(csv.DictReader(f))

    scores = [float(r["score"]) for r in rows]
    cids   = [r["candidate_id"] for r in rows]

    check("Output has 10 rows", len(rows) == 10)
    check("Scores monotonically non-increasing", all(scores[i] >= scores[i+1] for i in range(len(scores)-1)))
    check("Honeypot not in top-10", "CAND_HP_001" not in cids)
    check("Ranks are 1–10 sequential", [int(r["rank"]) for r in rows] == list(range(1, 11)))
    check("Reasoning column present and non-empty", all(r["reasoning"] for r in rows))
except Exception as e:
    check(f"End-to-end run crashed: {e}", False)
finally:
    os.unlink(tmp_in)
    if os.path.exists(tmp_out):
        os.unlink(tmp_out)

# ── Summary ───────────────────────────────────────────────────────
passed = sum(1 for _, s in results if s == PASS)
failed = sum(1 for _, s in results if s == FAIL)
print(f"\n── Results: {passed} passed, {failed} failed ──────────────────────────")
if failed == 0:
    print("🎉 All tests passed!\n")
else:
    print("⚠️  Some tests failed — check above.\n")
    sys.exit(1)