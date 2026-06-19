#!/usr/bin/env python3
"""
Redrob India Runs — Intelligent Candidate Discovery & Ranking
Author: Dhruv Bansal (Solo)

Approach: Multi-signal rule-based ranker that reasons like a senior recruiter,
not a keyword matcher. Five scoring pillars combined with a behavioral multiplier.
Honeypot detection built-in. Runs in <5 min on CPU with 16 GB RAM.
"""

import json
import csv
import argparse
import time
from datetime import date
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Constants — derived from close reading of the JD
# ─────────────────────────────────────────────────────────────────

# Reference date for "days since last active" calculations
TODAY = date(2026, 6, 18)

# ── Core technical skills the JD REQUIRES ────────────────────────
# These are the hard requirements: embeddings, vector DBs, Python, eval frameworks
CORE_SKILL_TOKENS = {
    # Embeddings / semantic retrieval
    "embedding", "embeddings", "sentence-transformer", "sentence_transformer",
    "bge", "e5 model", "openai embedding", "dense retrieval", "bi-encoder",
    "cross-encoder", "semantic search", "dense vector",
    # Vector DBs / hybrid search infra
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector database", "vector db", "vector store",
    "hybrid search", "ann", "approximate nearest neighbor",
    # Ranking / retrieval eval
    "ndcg", "mrr", "mean reciprocal rank", "map score", "mean average precision",
    "learning to rank", "ltr", "lambdamart", "ranknet", "information retrieval",
    "reranking", "reranker", "ranking system",
    # NLP / LLM (core domain)
    "nlp", "natural language processing", "transformers", "bert", "roberta",
    "llm", "large language model", "rag", "retrieval augmented",
    "text classification", "named entity", "question answering",
    # Production ML (shows shipping experience)
    "mlflow", "mlops", "model serving", "triton", "torchserve",
    "feature store", "online serving", "ab testing", "a/b testing",
}

# Skills that boost score but aren't required
NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "sft",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "distributed training", "ray", "dask",
    "recommendation system", "recommender system",
    "search relevance", "query understanding",
    "pytorch", "tensorflow", "jax",
}

# Primary specialisations that disqualify without NLP/IR overlap
DISQUALIFYING_DOMAIN_TOKENS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "resnet", "vgg",
    "speech recognition", "asr", "text-to-speech", "tts",
    "robotics", "ros framework", "autonomous driving",
}

# Consulting firms — purely services background is a red flag per JD
CONSULTING_FIRMS = {
    "tata consultancy", "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl technologies",
    "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    "persistent systems", "zensar", "niit technologies",
}

# Title patterns that signal relevance to this role
STRONG_TITLE_TOKENS = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied scientist", "applied ml", "research engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "retrieval", "data scientist",
    "senior software engineer", "senior engineer",
}

# Title patterns that strongly signal wrong role
WRONG_TITLE_TOKENS = {
    "marketing manager", "hr manager", "human resources",
    "content writer", "seo specialist", "graphic designer",
    "ux designer", "ui designer", "sales executive",
    "project manager", "program manager", "scrum master",
    "finance manager", "accountant", "operations manager",
    "business development", "product marketing",
}

# Tier-1 Indian cities the JD mentions or implies
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "bengaluru", "bangalore",
    "mumbai", "delhi", "gurgaon", "gurugram", "chennai",
    "kolkata", "ahmedabad",
}


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def days_ago(date_str: str) -> int:
    """Days between a YYYY-MM-DD date string and TODAY."""
    try:
        d = date.fromisoformat(date_str)
        return (TODAY - d).days
    except Exception:
        return 9999


def text_contains(text: str, tokens: set) -> int:
    """Count how many tokens from the set appear in text."""
    text_lower = text.lower()
    return sum(1 for t in tokens if t in text_lower)


def build_candidate_text(candidate: dict) -> str:
    """
    Concatenate all free-text fields so we can do token matching.
    This catches skill/experience described in career descriptions but
    not listed explicitly in the skills array (the 'plain-language Tier 5' trap).
    """
    parts = []
    profile = candidate.get("profile", {})
    parts.append(profile.get("summary", ""))
    parts.append(profile.get("headline", ""))

    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
        parts.append(job.get("company", ""))

    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))

    for cert in candidate.get("certifications", []):
        parts.append(cert.get("name", ""))
        parts.append(cert.get("issuer", ""))

    return " ".join(parts).lower()


# ─────────────────────────────────────────────────────────────────
# Honeypot Detection
# ─────────────────────────────────────────────────────────────────

def is_honeypot(candidate: dict) -> bool:
    """
    Detect subtly impossible profiles. Returns True if flagged.
    Honeypots include:
     - Many 'expert' skills with 0 months duration
     - Career history duration far shorter than claimed years_of_experience
     - Skills claiming expertise with impossibly short duration
    """
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])

    # Rule 1: ≥5 skills marked expert/advanced but 0 months duration
    zero_dur_expert = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 1) == 0
    )
    if zero_dur_expert >= 5:
        return True

    # Rule 2: "expert" skill with <3 months duration (extremely suspicious)
    impossible_expert = sum(
        1 for s in skills
        if s.get("proficiency") == "expert"
        and 0 < s.get("duration_months", 99) < 3
    )
    if impossible_expert >= 3:
        return True

    # Rule 3: Total career months significantly less than claimed YOE
    total_career_months = sum(j.get("duration_months", 0) for j in career)
    yoe = profile.get("years_of_experience", 0)
    if yoe > 2 and total_career_months < (yoe * 12 * 0.4):
        return True

    # Rule 4: Too many skills for believable breadth (>20 skills, many at expert)
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 10:
        return True

    return False


# ─────────────────────────────────────────────────────────────────
# Scoring Pillars
# ─────────────────────────────────────────────────────────────────

def score_technical_skills(candidate: dict, full_text: str) -> float:
    """
    Pillar 1: How well do the candidate's skills match JD requirements?
    Uses both the skills array (with proficiency/duration weighting) AND
    the full free-text corpus to catch 'plain-language Tier 5' candidates.

    Returns a float in [0, 1].
    """
    skills = candidate.get("skills", [])
    proficiency_weight = {"expert": 1.0, "advanced": 0.75, "intermediate": 0.45, "beginner": 0.15}

    # ── Weighted skills array score ──────────────────────────────
    weighted_core = 0.0
    for s in skills:
        name = s["name"].lower()
        prof = proficiency_weight.get(s.get("proficiency", "beginner"), 0.15)
        # cap duration at 72 months (6 years) to avoid a single skill dominating
        dur_months = min(s.get("duration_months", 0), 72)
        dur_factor = 1.0 + (dur_months / 72.0)  # range: 1.0 – 2.0

        # Check if this skill hits a core token
        if any(token in name for token in CORE_SKILL_TOKENS):
            weighted_core += prof * dur_factor

    # Normalise: expect a strong candidate to have ~8 weighted core matches
    weighted_score = min(weighted_core / 12.0, 1.0)

    # ── Free-text token coverage score ─────────────────────────
    # Catches experience described in career descriptions that isn't in skills
    text_core_hits = text_contains(full_text, CORE_SKILL_TOKENS)
    text_score = min(text_core_hits / 8.0, 1.0)

    # ── Nice-to-have bonus ──────────────────────────────────────
    nice_hits = text_contains(full_text, NICE_SKILLS)
    nice_score = min(nice_hits / 4.0, 1.0) * 0.2

    # ── Disqualifying primary domain penalty ────────────────────
    disq_hits = text_contains(full_text, DISQUALIFYING_DOMAIN_TOKENS)
    # Only penalise if disqualifying domain dominates AND no NLP/IR present
    nlp_hits = text_contains(full_text, {"nlp", "natural language", "retrieval", "ranking", "embedding"})
    disq_penalty = max(0, (disq_hits - nlp_hits) * 0.08)

    raw = 0.45 * weighted_score + 0.40 * text_score + 0.15 * nice_score
    return max(0.0, min(raw - disq_penalty, 1.0))


def score_career_trajectory(candidate: dict, full_text: str) -> float:
    """
    Pillar 2: Career quality — product companies, relevant titles, shipped systems.
    This is the strongest guard against keyword stuffers (wrong title = near-zero).

    Returns a float in [0, 1].
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = profile.get("current_title", "").lower()

    # ── Hard gate: obviously wrong current title ─────────────────
    for wrong in WRONG_TITLE_TOKENS:
        if wrong in current_title:
            return 0.04   # Near-zero; keyword stuffers can't pass this

    # ── Title relevance score ────────────────────────────────────
    title_score = 0.25  # baseline for ambiguous titles (e.g. "Software Engineer")
    for strong in STRONG_TITLE_TOKENS:
        if strong in current_title:
            title_score = 0.75
            break
    # Extra boost for exact target titles
    for exact in ("ml engineer", "ai engineer", "nlp engineer", "search engineer",
                  "recommendation", "retrieval", "ranking engineer"):
        if exact in current_title:
            title_score = 1.0
            break

    # ── Product vs consulting ratio ──────────────────────────────
    total_months = 0
    consulting_months = 0
    for job in career:
        co_lower = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        if any(c in co_lower for c in CONSULTING_FIRMS):
            consulting_months += dur

    consulting_ratio = (consulting_months / total_months) if total_months > 0 else 0
    # JD says "entire career at consulting = we don't move forward"
    if consulting_ratio >= 0.95:
        product_score = 0.05
    else:
        product_score = 1.0 - (consulting_ratio * 0.70)

    # ── Relevant role descriptions ───────────────────────────────
    # Score based on career descriptions mentioning production ML work
    production_signals = {
        "shipped", "deployed", "production", "serving", "latency",
        "a/b test", "ab test", "experiment", "offline eval", "online eval",
        "ndcg", "mrr", "precision@", "recall@", "benchmark",
        "embedding", "retrieval", "ranking", "recommendation",
        "vector", "index", "rerank",
    }
    desc_hits = 0
    for job in career[:4]:  # focus on recent roles
        desc = job.get("description", "").lower()
        desc_hits += sum(1 for sig in production_signals if sig in desc)

    desc_score = min(desc_hits / 10.0, 1.0)

    # ── Job-hopping penalty ──────────────────────────────────────
    # JD says: "optimize for titles by switching every 1.5 years = red flag"
    short_tenures = sum(1 for j in career if j.get("duration_months", 24) < 12)
    hopping_penalty = min(short_tenures * 0.06, 0.25)

    raw = (0.35 * title_score + 0.35 * product_score + 0.30 * desc_score) - hopping_penalty
    return max(0.0, min(raw, 1.0))


def score_experience_band(candidate: dict) -> float:
    """
    Pillar 3: Experience band match.
    JD says 5–9 years, tilts toward 6–8. Outside the band = lower score.

    Returns a float in [0, 1].
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    if 6.0 <= yoe <= 8.0:   # sweet spot ("ideal candidate" per JD)
        return 1.00
    elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0:
        return 0.88
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0:
        return 0.65
    elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 13.0:
        return 0.40
    elif yoe >= 13.0:
        return 0.25   # Too senior, likely architecture-only
    else:
        return 0.10   # Too junior


def score_location_availability(candidate: dict) -> float:
    """
    Pillar 4: Location fit.
    Pune/Noida are primary; other Tier-1 India cities are fine.
    Non-India candidates must be willing to relocate (JD says no visa sponsorship).

    Returns a float in [0, 1].
    """
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_relocate = signals.get("willing_to_relocate", False)

    # Tier-1 India cities — full score
    if country == "india":
        if any(city in location for city in ("pune", "noida", "bengaluru", "bangalore")):
            return 1.00
        elif any(city in location for city in PREFERRED_LOCATIONS):
            return 0.90
        else:
            return 0.75 if willing_relocate else 0.60

    # Outside India
    if willing_relocate:
        return 0.35   # JD: "case-by-case, no visa sponsorship"
    else:
        return 0.15


def score_behavioral_signals(candidate: dict) -> float:
    """
    Pillar 5: Platform behavioral signals — availability, responsiveness, engagement.
    Per JD: "perfect-on-paper candidate who hasn't logged in for 6 months and
    has a 5% recruiter response rate is, for hiring purposes, not actually available."

    Returns a float in [0, 1].
    """
    sig = candidate.get("redrob_signals", {})

    # Recency of activity
    inactive_days = days_ago(sig.get("last_active_date", "2020-01-01"))
    if inactive_days <= 14:
        recency = 1.00
    elif inactive_days <= 45:
        recency = 0.85
    elif inactive_days <= 90:
        recency = 0.65
    elif inactive_days <= 180:
        recency = 0.35
    else:
        recency = 0.08   # effectively unavailable

    # Open to work
    open_to_work = 1.0 if sig.get("open_to_work_flag", False) else 0.40

    # Recruiter response rate (key signal per signals doc)
    response_rate = float(sig.get("recruiter_response_rate", 0.0))

    # Notice period (JD strongly prefers ≤30 days)
    notice = sig.get("notice_period_days", 90)
    if notice <= 15:
        notice_score = 1.00
    elif notice <= 30:
        notice_score = 0.90
    elif notice <= 60:
        notice_score = 0.65
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.20

    # Interview completion rate
    interview_rate = float(sig.get("interview_completion_rate", 0.5))

    # GitHub activity — relevant for an AI engineer role
    github = sig.get("github_activity_score", -1)
    github_score = (github / 100.0) if github >= 0 else 0.25

    # Profile completeness (signal of seriousness)
    completeness = sig.get("profile_completeness_score", 50) / 100.0

    behavioral = (
        0.28 * recency
        + 0.22 * open_to_work
        + 0.20 * response_rate
        + 0.15 * notice_score
        + 0.08 * interview_rate
        + 0.04 * github_score
        + 0.03 * completeness
    )
    return max(0.0, min(behavioral, 1.0))


# ─────────────────────────────────────────────────────────────────
# Composite Scorer
# ─────────────────────────────────────────────────────────────────

# Weights tuned to match JD priorities:
# Skills + Career dominate; behavioral acts as a multiplier at the margin
WEIGHTS = {
    "skills":    0.32,
    "career":    0.28,
    "experience":0.18,
    "behavioral":0.14,
    "location":  0.08,
}

def compute_score(candidate: dict) -> tuple[float, dict]:
    """
    Compute composite score for one candidate.
    Returns (total_score, breakdown_dict).
    """
    if is_honeypot(candidate):
        return 0.001, {"honeypot": True}

    full_text = build_candidate_text(candidate)

    skills_s   = score_technical_skills(candidate, full_text)
    career_s   = score_career_trajectory(candidate, full_text)
    exp_s      = score_experience_band(candidate)
    behav_s    = score_behavioral_signals(candidate)
    loc_s      = score_location_availability(candidate)

    total = (
        WEIGHTS["skills"]     * skills_s
        + WEIGHTS["career"]   * career_s
        + WEIGHTS["experience"] * exp_s
        + WEIGHTS["behavioral"] * behav_s
        + WEIGHTS["location"] * loc_s
    )

    breakdown = {
        "skills": round(skills_s, 3),
        "career": round(career_s, 3),
        "experience": round(exp_s, 3),
        "behavioral": round(behav_s, 3),
        "location": round(loc_s, 3),
        "total": round(total, 4),
    }
    return round(total, 6), breakdown


# ─────────────────────────────────────────────────────────────────
# Reasoning Generator
# ─────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, breakdown: dict, rank: int) -> str:
    """
    Generate a 1-2 sentence reasoning specific to this candidate.
    Per Stage 4 rubric: must cite specific facts, connect to JD,
    acknowledge concerns, avoid hallucination, and match rank tone.
    """
    if breakdown.get("honeypot"):
        return "Flagged as invalid profile — honeypot pattern detected."

    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Pull actual skill names that match core requirements
    proficiency_weight = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
    core_skills_found = sorted(
        [
            s for s in skills
            if any(t in s["name"].lower() for t in CORE_SKILL_TOKENS)
        ],
        key=lambda s: proficiency_weight.get(s.get("proficiency", "beginner"), 0),
        reverse=True
    )[:3]
    skill_str = ", ".join(s["name"] for s in core_skills_found) if core_skills_found else None

    # Behavioral quick facts
    inactive_days = days_ago(sig.get("last_active_date", "2020-01-01"))
    response_rate = sig.get("recruiter_response_rate", 0)
    notice = sig.get("notice_period_days", 90)
    open_flag = sig.get("open_to_work_flag", False)

    # Compose reason sentences
    parts = []

    # Sentence 1: who they are + key strength
    if skill_str:
        parts.append(
            f"{title} with {yoe:.1f} yrs at {company}; "
            f"core skills: {skill_str}."
        )
    else:
        parts.append(
            f"{title} with {yoe:.1f} yrs at {company}; "
            f"skills match JD requirements for retrieval/ranking engineering."
        )

    # Sentence 2: availability + concerns (honest, rank-appropriate)
    avail_notes = []
    if open_flag and inactive_days <= 45:
        avail_notes.append("actively available")
    elif inactive_days > 180:
        avail_notes.append(f"inactive for {inactive_days}d — availability uncertain")
    else:
        avail_notes.append(f"last active {inactive_days}d ago")

    avail_notes.append(f"response rate {response_rate:.0%}")
    avail_notes.append(f"{notice}d notice")

    # Concerns based on score breakdown and rank
    concerns = []
    if breakdown.get("career", 1) < 0.30:
        concerns.append("limited product-company ML history")
    if breakdown.get("skills", 1) < 0.25:
        concerns.append("partial skill overlap with JD requirements")
    if breakdown.get("experience", 1) < 0.50:
        concerns.append(
            "experience outside 5–9 yr target band"
            if yoe < 5 or yoe > 9 else ""
        )
    if country.lower() != "india":
        concerns.append("non-India location (no visa sponsorship per JD)")

    concern_str = "; ".join(c for c in concerns if c)

    avail_str = ", ".join(avail_notes)
    if concern_str and rank > 20:
        parts.append(f"Signals: {avail_str}. Concern: {concern_str}.")
    else:
        parts.append(f"Signals: {avail_str}.")

    return " ".join(parts)[:500]


# ─────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument(
        "--candidates",
        default="candidates.jsonl",
        help="Path to candidates.jsonl (or .jsonl.gz)"
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=100,
        help="Number of candidates to include in output"
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    # ── Load candidates ──────────────────────────────────────────
    t0 = time.time()
    print(f"[rank.py] Loading candidates from {candidates_path} …")

    candidates = []
    if str(candidates_path).endswith(".gz"):
        import gzip
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(candidates_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))

    print(f"[rank.py] Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # ── Score all candidates ─────────────────────────────────────
    print("[rank.py] Scoring …")
    t1 = time.time()

    scored = []  # (score, candidate_id, breakdown, candidate)
    honeypot_count = 0

    for i, c in enumerate(candidates):
        if i % 20_000 == 0 and i > 0:
            elapsed = time.time() - t1
            eta = elapsed / i * (len(candidates) - i)
            print(f"  {i:,}/{len(candidates):,} — {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

        score, bd = compute_score(c)
        if bd.get("honeypot"):
            honeypot_count += 1
        scored.append((score, c["candidate_id"], bd, c))

    elapsed_scoring = time.time() - t1
    print(f"[rank.py] Scored {len(candidates):,} in {elapsed_scoring:.1f}s")
    print(f"[rank.py] Honeypots detected: {honeypot_count}")

    # ── Sort: score DESC, candidate_id ASC (tie-break per spec) ──
    scored.sort(key=lambda x: (-x[0], x[1]))

    # ── Write top-N CSV ──────────────────────────────────────────
    top_n = scored[:args.topn]
    out_path = Path(args.out)

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, bd, candidate) in enumerate(top_n, start=1):
            reasoning = generate_reasoning(candidate, bd, rank)
            writer.writerow([cid, rank, f"{score:.6f}", reasoning])

    total_time = time.time() - t0
    print(f"\n[rank.py] ✓ Submission written → {out_path}")
    print(f"[rank.py] Total wall-clock time: {total_time:.1f}s")

    # ── Print top-10 preview ─────────────────────────────────────
    print("\n── Top 10 Preview ──────────────────────────────────")
    for rank, (score, cid, bd, c) in enumerate(top_n[:10], start=1):
        p = c["profile"]
        print(
            f"  #{rank:>3}  {cid}  score={score:.4f}  "
            f"{p['current_title'][:35]:<35}  "
            f"YOE={p['years_of_experience']:.1f}  "
            f"{p['location']}, {p['country']}"
        )
        print(f"         skills={bd['skills']:.2f}  "
              f"career={bd['career']:.2f}  "
              f"exp={bd['experience']:.2f}  "
              f"behav={bd['behavioral']:.2f}  "
              f"loc={bd['location']:.2f}")


if __name__ == "__main__":
    main()
