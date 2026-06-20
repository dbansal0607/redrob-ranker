#!/usr/bin/env python3
"""
Redrob India Runs — Intelligent Candidate Discovery & Ranking
Author: Dhruv Bansal (Solo)

Approach: Multi-signal rule-based ranker that reasons like a senior recruiter.
Five scoring pillars + behavioral multiplier + honeypot detection.
Multiprocessing-enabled. Config-driven weights. Structured logging.
Runs in <5 min on CPU with 16 GB RAM. Zero external dependencies.
"""

import json
import csv
import argparse
import logging
import time
import multiprocessing as mp
from datetime import date
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("redrob-ranker")

# ── Reference date ────────────────────────────────────────────────
TODAY = date(2026, 6, 18)

# ─────────────────────────────────────────────────────────────────
# Constants — derived from JD analysis
# ─────────────────────────────────────────────────────────────────

CORE_SKILL_TOKENS = {
    # Embeddings / semantic retrieval
    "embedding", "embeddings", "sentence-transformer", "sentence_transformer",
    "bge", "e5 model", "openai embedding", "dense retrieval", "bi-encoder",
    "cross-encoder", "semantic search", "dense vector", "text embeddings",
    "cohere", "ada-002", "text-embedding",
    # Vector DBs / ANN infra
    "pinecone", "weaviate", "qdrant", "milvus", "faiss", "opensearch",
    "elasticsearch", "vector database", "vector db", "vector store",
    "hybrid search", "ann", "approximate nearest neighbor", "hnsw",
    "chroma", "chromadb", "vespa", "typesense", "pgvector",
    # Ranking / retrieval eval
    "ndcg", "mrr", "mean reciprocal rank", "map score", "mean average precision",
    "learning to rank", "ltr", "lambdamart", "ranknet", "ranklib",
    "information retrieval", "reranking", "reranker", "ranking system",
    "bm25", "tf-idf", "tfidf", "sparse retrieval", "dense retrieval",
    "two-stage retrieval", "recall@", "precision@",
    # NLP / LLM core
    "nlp", "natural language processing", "transformers", "bert", "roberta",
    "llm", "large language model", "rag", "retrieval augmented",
    "text classification", "named entity", "question answering",
    "gpt", "claude", "gemini", "mistral", "llama", "falcon",
    "prompt engineering", "langchain", "llamaindex",
    # Production ML engineering
    "mlflow", "mlops", "model serving", "triton", "torchserve", "bentoml",
    "feature store", "online serving", "ab testing", "a/b testing",
    "shadow deployment", "canary", "drift detection",
    # Python / engineering core
    "python", "fastapi", "flask", "celery", "redis", "kafka",
    "spark", "airflow", "dbt", "sql", "postgresql",
}

NICE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "sft", "rlhf",
    "xgboost", "lightgbm", "catboost", "gradient boosting",
    "distributed training", "ray", "dask", "horovod",
    "recommendation system", "recommender system",
    "search relevance", "query understanding", "query expansion",
    "pytorch", "tensorflow", "jax", "triton inference",
    "knowledge graph", "neo4j", "graph neural",
}

DISQUALIFYING_DOMAIN_TOKENS = {
    "computer vision", "image classification", "object detection",
    "image segmentation", "yolo", "resnet", "vgg", "convolutional",
    "speech recognition", "asr", "text-to-speech", "tts", "whisper",
    "robotics", "ros framework", "autonomous driving", "slam",
}

CONSULTING_FIRMS = {
    "tata consultancy", "tcs", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "tech mahindra", "hcl technologies",
    "mphasis", "hexaware", "l&t infotech", "ltimindtree",
    "persistent systems", "zensar", "niit technologies",
    "mindtree", "mastech", "syntel", "kpit",
}

STRONG_TITLE_TOKENS = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied scientist", "applied ml", "research engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation", "retrieval", "data scientist",
    "senior software engineer", "senior engineer", "staff engineer",
    "principal engineer", "founding engineer",
}

WRONG_TITLE_TOKENS = {
    "marketing manager", "hr manager", "human resources",
    "content writer", "seo specialist", "graphic designer",
    "ux designer", "ui designer", "sales executive",
    "project manager", "program manager", "scrum master",
    "finance manager", "accountant", "operations manager",
    "business development", "product marketing", "recruiter",
    "talent acquisition",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "bengaluru", "bangalore",
    "mumbai", "delhi", "gurgaon", "gurugram", "chennai",
    "kolkata", "ahmedabad", "ncr",
}

# ─────────────────────────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load scoring weights from config.yaml if available, else use defaults."""
    defaults = {
        "scoring_weights": {
            "skills": 0.32, "career": 0.28,
            "experience": 0.18, "behavioral": 0.14, "location": 0.08
        }
    }
    try:
        import re
        with open(config_path) as f:
            content = f.read()
        # Simple YAML key: value parser (no external deps)
        weights = {}
        in_weights = False
        for line in content.splitlines():
            if "scoring_weights:" in line:
                in_weights = True
                continue
            if in_weights:
                m = re.match(r"\s+(\w+):\s*([\d.]+)", line)
                if m:
                    weights[m.group(1)] = float(m.group(2))
                elif line and not line.startswith(" "):
                    break
        if weights:
            defaults["scoring_weights"] = weights
            log.info(f"Loaded config from {config_path}: {weights}")
    except FileNotFoundError:
        log.info("config.yaml not found — using default weights")
    except Exception as e:
        log.warning(f"Config parse error: {e} — using defaults")
    return defaults

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def days_ago(date_str: str) -> int:
    try:
        return (TODAY - date.fromisoformat(date_str)).days
    except Exception:
        return 9999

def text_contains(text: str, tokens: set) -> int:
    text_lower = text.lower()
    return sum(1 for t in tokens if t in text_lower)

def build_candidate_text(candidate: dict) -> str:
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
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    # Rule 1: ≥5 expert/advanced skills with 0 months
    if sum(1 for s in skills
           if s.get("proficiency") in ("expert", "advanced")
           and s.get("duration_months", 1) == 0) >= 5:
        return True

    # Rule 2: ≥3 expert skills with <3 months
    if sum(1 for s in skills
           if s.get("proficiency") == "expert"
           and 0 < s.get("duration_months", 99) < 3) >= 3:
        return True

    # Rule 3: Career months << claimed YOE
    total_months = sum(j.get("duration_months", 0) for j in career)
    if yoe > 2 and total_months < (yoe * 12 * 0.4):
        return True

    # Rule 4: ≥10 expert skills (implausible breadth)
    if sum(1 for s in skills if s.get("proficiency") == "expert") >= 10:
        return True

    return False

# ─────────────────────────────────────────────────────────────────
# Scoring Pillars
# ─────────────────────────────────────────────────────────────────

def score_technical_skills(candidate: dict, full_text: str) -> float:
    skills = candidate.get("skills", [])
    proficiency_weight = {"expert": 1.0, "advanced": 0.75,
                          "intermediate": 0.45, "beginner": 0.15}

    # Weighted skills array
    weighted_core = 0.0
    for s in skills:
        name = s["name"].lower()
        prof = proficiency_weight.get(s.get("proficiency", "beginner"), 0.15)
        dur_months = min(s.get("duration_months", 0), 72)
        dur_factor = 1.0 + (dur_months / 72.0)
        if any(token in name for token in CORE_SKILL_TOKENS):
            weighted_core += prof * dur_factor

    weighted_score = min(weighted_core / 12.0, 1.0)

    # Free-text corpus score
    text_core_hits = text_contains(full_text, CORE_SKILL_TOKENS)
    text_score = min(text_core_hits / 8.0, 1.0)

    # Nice-to-have bonus
    nice_hits = text_contains(full_text, NICE_SKILLS)
    nice_score = min(nice_hits / 4.0, 1.0) * 0.2

    # Disqualifying domain penalty (only if no NLP/IR overlap)
    disq_hits = text_contains(full_text, DISQUALIFYING_DOMAIN_TOKENS)
    nlp_hits = text_contains(full_text, {
        "nlp", "natural language", "retrieval", "ranking", "embedding"
    })
    disq_penalty = max(0, (disq_hits - nlp_hits) * 0.08)

    raw = 0.45 * weighted_score + 0.40 * text_score + 0.15 * nice_score
    return max(0.0, min(raw - disq_penalty, 1.0))


def score_career_trajectory(candidate: dict, full_text: str) -> float:
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    current_title = profile.get("current_title", "").lower()

    # Hard gate: wrong title
    for wrong in WRONG_TITLE_TOKENS:
        if wrong in current_title:
            return 0.04

    # Title relevance
    title_score = 0.25
    for strong in STRONG_TITLE_TOKENS:
        if strong in current_title:
            title_score = 0.75
            break
    for exact in ("ml engineer", "ai engineer", "nlp engineer",
                  "search engineer", "recommendation", "retrieval",
                  "ranking engineer", "applied scientist", "staff engineer"):
        if exact in current_title:
            title_score = 1.0
            break

    # Product vs consulting ratio
    total_months, consulting_months = 0, 0
    for job in career:
        co = job.get("company", "").lower()
        dur = job.get("duration_months", 0)
        total_months += dur
        if any(c in co for c in CONSULTING_FIRMS):
            consulting_months += dur

    consulting_ratio = (consulting_months / total_months) if total_months > 0 else 0
    product_score = 0.05 if consulting_ratio >= 0.95 else 1.0 - (consulting_ratio * 0.70)

    # Production ML in descriptions
    production_signals = {
        "shipped", "deployed", "production", "serving", "latency",
        "a/b test", "ab test", "experiment", "offline eval", "online eval",
        "ndcg", "mrr", "precision@", "recall@", "benchmark",
        "embedding", "retrieval", "ranking", "recommendation",
        "vector", "index", "rerank", "feature store", "pipeline",
    }
    desc_hits = sum(
        sum(1 for sig in production_signals if sig in job.get("description", "").lower())
        for job in career[:4]
    )
    desc_score = min(desc_hits / 10.0, 1.0)

    # Job-hopping penalty
    short_tenures = sum(1 for j in career if j.get("duration_months", 24) < 12)
    hopping_penalty = min(short_tenures * 0.06, 0.25)

    raw = (0.35 * title_score + 0.35 * product_score + 0.30 * desc_score) - hopping_penalty
    return max(0.0, min(raw, 1.0))


def score_experience_band(candidate: dict) -> float:
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    if 6.0 <= yoe <= 8.0:   return 1.00
    elif 5.0 <= yoe < 6.0 or 8.0 < yoe <= 9.0: return 0.88
    elif 4.0 <= yoe < 5.0 or 9.0 < yoe <= 11.0: return 0.65
    elif 3.0 <= yoe < 4.0 or 11.0 < yoe <= 13.0: return 0.40
    elif yoe >= 13.0: return 0.25
    else: return 0.10


def score_location_availability(candidate: dict) -> float:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)

    if country == "india":
        if any(city in location for city in ("pune", "noida", "bengaluru", "bangalore")):
            return 1.00
        elif any(city in location for city in PREFERRED_LOCATIONS):
            return 0.90
        return 0.75 if willing else 0.60
    return 0.35 if willing else 0.15


def score_behavioral_signals(candidate: dict) -> float:
    sig = candidate.get("redrob_signals", {})

    inactive_days = days_ago(sig.get("last_active_date", "2020-01-01"))
    if inactive_days <= 14:   recency = 1.00
    elif inactive_days <= 45: recency = 0.85
    elif inactive_days <= 90: recency = 0.65
    elif inactive_days <= 180:recency = 0.35
    else:                     recency = 0.08

    open_to_work   = 1.0 if sig.get("open_to_work_flag", False) else 0.40
    response_rate  = float(sig.get("recruiter_response_rate", 0.0))
    notice         = sig.get("notice_period_days", 90)
    notice_score   = (1.00 if notice <= 15 else 0.90 if notice <= 30
                      else 0.65 if notice <= 60 else 0.45 if notice <= 90 else 0.20)
    interview_rate = float(sig.get("interview_completion_rate", 0.5))
    github         = sig.get("github_activity_score", -1)
    github_score   = (github / 100.0) if github >= 0 else 0.25
    completeness   = sig.get("profile_completeness_score", 50) / 100.0

    return max(0.0, min(
        0.28 * recency + 0.22 * open_to_work + 0.20 * response_rate
        + 0.15 * notice_score + 0.08 * interview_rate
        + 0.04 * github_score + 0.03 * completeness,
        1.0
    ))

# ─────────────────────────────────────────────────────────────────
# Composite Scorer
# ─────────────────────────────────────────────────────────────────

# Global weights (set at startup from config)
WEIGHTS = {"skills": 0.32, "career": 0.28,
           "experience": 0.18, "behavioral": 0.14, "location": 0.08}

def compute_score(candidate: dict) -> tuple:
    if is_honeypot(candidate):
        return 0.001, {"honeypot": True}

    full_text = build_candidate_text(candidate)
    skills_s  = score_technical_skills(candidate, full_text)
    career_s  = score_career_trajectory(candidate, full_text)
    exp_s     = score_experience_band(candidate)
    behav_s   = score_behavioral_signals(candidate)
    loc_s     = score_location_availability(candidate)

    total = (WEIGHTS["skills"]     * skills_s
           + WEIGHTS["career"]     * career_s
           + WEIGHTS["experience"] * exp_s
           + WEIGHTS["behavioral"] * behav_s
           + WEIGHTS["location"]   * loc_s)

    return round(total, 6), {
        "skills": round(skills_s, 3), "career": round(career_s, 3),
        "experience": round(exp_s, 3), "behavioral": round(behav_s, 3),
        "location": round(loc_s, 3), "total": round(total, 4),
    }

# ─────────────────────────────────────────────────────────────────
# Multiprocessing worker
# ─────────────────────────────────────────────────────────────────

def _score_worker(candidate: dict) -> tuple:
    """Worker function for multiprocessing pool."""
    try:
        score, bd = compute_score(candidate)
        return (score, candidate["candidate_id"], bd, candidate)
    except Exception as e:
        cid = candidate.get("candidate_id", "UNKNOWN")
        log.warning(f"Error scoring {cid}: {e}")
        return (0.0, cid, {"error": str(e)}, candidate)

# ─────────────────────────────────────────────────────────────────
# Reasoning Generator
# ─────────────────────────────────────────────────────────────────

def generate_reasoning(candidate: dict, breakdown: dict, rank: int) -> str:
    if breakdown.get("honeypot"):
        return "Flagged as invalid profile — honeypot pattern detected."

    profile = candidate.get("profile", {})
    sig     = candidate.get("redrob_signals", {})
    skills  = candidate.get("skills", [])

    title    = profile.get("current_title", "Unknown")
    yoe      = profile.get("years_of_experience", 0)
    company  = profile.get("current_company", "")
    country  = profile.get("country", "")

    prof_w   = {"expert": 3, "advanced": 2, "intermediate": 1, "beginner": 0}
    core_skills = sorted(
        [s for s in skills if any(t in s["name"].lower() for t in CORE_SKILL_TOKENS)],
        key=lambda s: prof_w.get(s.get("proficiency", "beginner"), 0),
        reverse=True
    )[:3]
    skill_str = ", ".join(s["name"] for s in core_skills) if core_skills else None

    inactive_days = days_ago(sig.get("last_active_date", "2020-01-01"))
    response_rate = sig.get("recruiter_response_rate", 0)
    notice        = sig.get("notice_period_days", 90)
    open_flag     = sig.get("open_to_work_flag", False)

    parts = []
    if skill_str:
        parts.append(f"{title} with {yoe:.1f} yrs at {company}; core skills: {skill_str}.")
    else:
        parts.append(f"{title} with {yoe:.1f} yrs at {company}.")

    avail = []
    if open_flag and inactive_days <= 45:
        avail.append("actively available")
    elif inactive_days > 180:
        avail.append(f"inactive {inactive_days}d — availability uncertain")
    else:
        avail.append(f"last active {inactive_days}d ago")
    avail += [f"response rate {response_rate:.0%}", f"{notice}d notice"]

    concerns = []
    if breakdown.get("career", 1) < 0.30: concerns.append("limited product-company ML history")
    if breakdown.get("skills", 1) < 0.25: concerns.append("partial skill overlap with JD")
    if country.lower() != "india":         concerns.append("non-India location")

    parts.append(f"Signals: {', '.join(avail)}." +
                 (f" Concern: {'; '.join(concerns)}." if concerns and rank > 20 else ""))
    return " ".join(parts)[:500]

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out",        default="submission.csv")
    parser.add_argument("--topn",       type=int, default=100)
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--workers",    type=int,
                        default=max(1, mp.cpu_count() - 1),
                        help="Parallel workers (default: CPU count - 1)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    WEIGHTS.update(cfg.get("scoring_weights", {}))
    log.info(f"Weights: {WEIGHTS}")

    # Load candidates
    t0 = time.time()
    log.info(f"Loading candidates from {args.candidates} …")
    candidates = []
    path = Path(args.candidates)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    candidates.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed JSON line: {e}")

    log.info(f"Loaded {len(candidates):,} candidates in {time.time()-t0:.1f}s")

    # Score — multiprocessing if >10K candidates
    t1 = time.time()
    if len(candidates) > 10_000 and args.workers > 1:
        log.info(f"Scoring with {args.workers} workers …")
        with mp.Pool(args.workers) as pool:
            scored = pool.map(_score_worker, candidates)
    else:
        log.info("Scoring (single process) …")
        scored = [_score_worker(c) for c in candidates]

    honeypots = sum(1 for s in scored if s[2].get("honeypot"))
    errors    = sum(1 for s in scored if s[2].get("error"))
    log.info(f"Scored {len(scored):,} in {time.time()-t1:.1f}s | "
             f"honeypots={honeypots} | errors={errors}")

    # Sort & top-N
    scored.sort(key=lambda x: (-x[0], x[1]))
    top_n = scored[:args.topn]

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, bd, candidate) in enumerate(top_n, 1):
            writer.writerow([cid, rank, f"{score:.6f}",
                             generate_reasoning(candidate, bd, rank)])

    log.info(f"Submission written → {out_path} ({len(top_n)} candidates)")
    log.info(f"Total wall-clock: {time.time()-t0:.1f}s")

    # Top-10 preview
    log.info("── Top 10 ──────────────────────────────────")
    for rank, (score, cid, bd, c) in enumerate(top_n[:10], 1):
        p = c["profile"]
        log.info(f"#{rank:>2} {cid}  {score:.4f}  "
                 f"{p['current_title'][:30]:<30}  "
                 f"YOE={p['years_of_experience']:.1f}  "
                 f"{p['location']}, {p['country']}")

if __name__ == "__main__":
    main()