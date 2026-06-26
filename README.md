# Redrob India Runs — Intelligent Candidate Discovery & Ranking

**Submission by:** Dhruv Bansal (Solo)
**Hackathon:** Redrob India Runs — Data & AI Challenge
**GitHub:** https://github.com/dbansal0607/redrob-ranker

---

## Reproduce in Three Commands

```bash
pip install sentence-transformers numpy   # install dependencies (once)
python download_model.py                  # download model locally (once)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- Runtime: ~2.5 minutes on CPU (well within 5-minute limit)
- Memory: ~1.5 GB peak (well within 16 GB limit)
- Network: zero calls during ranking (model pre-downloaded)
- GPU: not used

---

## Architecture — Two-Stage Hybrid Ranker

```
[candidates.jsonl]
        ↓
[Stage 1: Rule Filter]          ~30s
100,000 → 2,000 candidates
Title gate + honeypot detection +
experience band + consulting penalty +
behavioral availability multiplier
        ↓
[Stage 2: Semantic Reranker]    ~90s
2,000 → top-100 candidates
all-MiniLM-L6-v2 bi-encoder
JD vs candidate career text cosine similarity
        ↓
[submission.csv + per-candidate reasoning]
```

**Final score = 0.45 x rule_score + 0.55 x semantic_score**

---

## Why Two Stages?

A rule-only ranker misses candidates who describe production ML work
in plain English without AI buzzwords. For example:

> "Built system finding similar documents using dense vector
> representations and approximate search indexes"

This candidate has FAISS + embedding experience but no keyword
matches. Stage 2 catches them via semantic similarity to the JD.

---

## Five Scoring Pillars (Stage 1)

| Pillar | Weight | What it measures |
|---|---|---|
| Technical Skills | 32% | Core skill match via skills array + career description text |
| Career Trajectory | 28% | Product company ratio, title fit, production ML signals, job-hopping penalty |
| Experience Band | 18% | 5-9 yr sweet spot (ideal: 6-8 per JD) |
| Behavioral Availability | 14% | Recency, response rate, notice period, open-to-work flag |
| Location | 8% | Pune/Noida/Tier-1 India priority |

---

## Key Design Decisions

**1. Word-boundary regex matching**
Single precompiled alternation pattern — prevents "rag" matching
"average", "storage", "fragment". O(n) over text instead of O(n x m).

**2. Hard career gate against keyword stuffers**
Wrong title (HR Manager, Marketing Manager etc.) → career score 0.04
regardless of how many AI skills are listed.

**3. Consulting ratio penalty**
TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini etc. — 95%+ consulting
career → product score 0.05.

**4. Neutral behavioral defaults**
Missing recruiter_response_rate defaults to 0.70, not 0.0 — protects
external candidates with no platform history.

**5. Honeypot detection (3 rules)**
- 5+ expert/advanced skills with 0 months duration
- 3+ expert skills with less than 3 months duration
- Career months less than 40% of claimed YOE

**6. Softened experience banding**
13+ YOE candidates score 0.35-0.40 instead of 0.25 — senior talent
is not disqualified, just slightly penalized for being overqualified.

---

## File Structure

```
redrob-ranker/
├── rank.py                  # Two-stage ranker — main entry point
├── download_model.py        # Run once to cache model locally
├── jd.txt                   # Job description for semantic embedding
├── config.yaml              # Scoring weights (edit without touching code)
├── requirements.txt         # sentence-transformers, numpy
├── test_rank.py             # 36 unit tests — all passing
├── validate_submission.py   # Official format validator
├── Dockerfile               # Containerized reproducible environment
└── README.md                # This file
```

---

## Validation

```bash
python validate_submission.py submission.csv
# Expected: "Submission is valid."
```

---

## Running Tests

```bash
python test_rank.py
# Expected: 36 passed, 0 failed — All tests passed!
```

---

## Docker (fully reproducible)

```bash
docker build -t redrob-ranker .
docker run -v $(pwd):/data redrob-ranker
```

Model is baked into the Docker image during build — no internet needed at runtime.

---

## Compute Environment

- Platform: Windows 11 / PowerShell
- Python: 3.12
- Dependencies: sentence-transformers, numpy
- CPU: multiprocessing enabled (auto-detects core count)
- RAM: ~1.5 GB peak
- GPU: not used
- Network during ranking: zero calls