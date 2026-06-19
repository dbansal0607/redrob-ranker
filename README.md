# Redrob India Runs — Intelligent Candidate Discovery & Ranking

**Submission by:** Dhruv Bansal (Solo)  
**Hackathon:** Redrob India Runs — Data & AI Challenge  
**GitHub:** https://github.com/dbansal0607/redrob-ranker

---

## Reproduce in One Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- Runtime: ~60 seconds on CPU (well within 5-minute limit)
- Memory: ~1.5 GB peak (well within 16 GB limit)
- Network: zero external calls
- GPU: not used

---

## Approach Overview

**Multi-signal rule-based ranker with explicit reasoning capture.**

Rather than pure keyword matching, this system reasons about *what the JD actually means* — reading career history descriptions, computing a product-company ratio, detecting title mismatches, and applying behavioral signals as a real-time availability multiplier.

### Five Scoring Pillars

| Pillar | Weight | What it measures |
|---|---|---|
| Technical Skills | 32% | Core skill match via both skills array + career description text |
| Career Trajectory | 28% | Product company ratio, title fit, production ML signals, job-hopping penalty |
| Experience Band | 18% | 5–9 yr sweet spot (ideal: 6–8 per JD's "ideal candidate" section) |
| Behavioral Availability | 14% | Recency, response rate, notice period, open-to-work flag |
| Location | 8% | Pune/Noida/Tier-1 India priority; non-India penalised (no visa sponsorship per JD) |

### Key Design Decisions

**1. Dual text corpus for skill detection**  
Skills array + full free-text (career descriptions + summary + headline). This catches "plain-language Tier 5" candidates who describe embedding-based retrieval work in job descriptions but don't use the exact buzzword in their skills list.

**2. Hard career gate against keyword stuffers**  
If the current title matches any WRONG_TITLE_TOKENS (marketing manager, HR manager, content writer, etc.), career score floors at 0.04 — making it nearly impossible for keyword stuffers to rank in the top 100 regardless of how many AI skills they list.

**3. Consulting ratio penalty**  
Career history is scanned for known IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.). A candidate with 95%+ of career at consulting firms receives a 0.05 product score per JD's explicit disqualifier.

**4. Behavioral multiplier anchored to real-time signals**  
Response rate, days-since-active, notice period, and open-to-work flag are combined into a behavioral score. A candidate inactive for 6+ months scores ~0.08 on recency alone, directly implementing the JD's note that such candidates are "not actually available for hiring purposes."

**5. Honeypot detection**  
Four heuristics catch impossible profiles:
- ≥5 skills at expert/advanced proficiency with 0 months duration
- ≥3 skills at expert proficiency with <3 months duration  
- Total career months < 40% of claimed YOE
- ≥10 expert-level skills (implausibly broad expertise)

Flagged candidates receive score ≈ 0.001 and never appear in the top 100.

---

## File Structure

```
redrob-ranker/
├── rank.py                       # Main ranker — single entry point
├── requirements.txt              # No external dependencies
├── submission_metadata.yaml      # Portal metadata (fill before uploading)
├── validate_submission.py        # Format validator (provided by organizers)
├── README.md                     # This file
└── submission.csv                # Generated output (not committed to repo)
```

---

## Running on a Small Sample (Sandbox)

For the sandbox/demo environment, you can run on a sample input:

```bash
# Use any subset of candidates.jsonl as input
head -n 1000 candidates.jsonl > sample_1000.jsonl
python rank.py --candidates sample_1000.jsonl --out sample_submission.csv --topn 100
```

Note: with <100 unique candidates in the pool, `--topn` is automatically capped.

---

## Validation

```bash
python validate_submission.py submission.csv
# Expected: "Submission is valid."
```

---

## Architecture Notes

- **Zero external dependencies**: stdlib only (`json`, `csv`, `argparse`, `datetime`, `pathlib`, `time`, `gzip`)
- **Streaming-friendly**: processes candidates one line at a time from JSONL
- Accepts both `.jsonl` and `.jsonl.gz` inputs
- All scoring is deterministic — same input always produces same output
- Tie-breaking follows spec: equal scores resolved by `candidate_id` ascending

---

## Compute Environment

- Platform: Windows 11 / WSL2 Ubuntu 24.04
- Python: 3.12
- CPU cores: available system cores (single-threaded scoring loop)
- RAM: ~1.5 GB peak usage
- GPU: not used
- Network during ranking: none
