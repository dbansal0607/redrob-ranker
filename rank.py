#!/usr/bin/env python3
"""
Redrob India Runs — Intelligent Candidate Discovery & Ranking
Author: Dhruv Bansal (Solo)

Two-Stage Hybrid Ranker:
  Stage 1 — Rule-based filter:  100,000 → 2,000 candidates (~30s)
  Stage 2 — Semantic reranker:  2,000   → 100 candidates   (~90s)

Final score = 0.45 * rule_score + 0.55 * semantic_score

Run download_model.py ONCE before ranking. Then zero network calls.

Requirements:
    pip install sentence-transformers numpy
    python download_model.py   # once only

Usage:
    python rank.py --candidates candidates.jsonl --out submission.csv
"""

import json, csv, argparse, logging, time, re, sys
import multiprocessing as mp
from datetime import date
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("redrob-ranker")

TODAY       = date(2026, 6, 18)
MODEL_DIR   = Path("./model")
JD_FILE     = Path("./jd.txt")
STAGE1_KEEP = 2000
RULE_W      = 0.45
SEM_W       = 0.55

# ── Token sets ────────────────────────────────────────────────────
CORE_SKILL_TOKENS = {
    "embedding","embeddings","sentence-transformer","sentence_transformer",
    "bge","e5 model","openai embedding","dense retrieval","bi-encoder",
    "cross-encoder","semantic search","dense vector","text embeddings",
    "cohere","ada-002","text-embedding",
    "pinecone","weaviate","qdrant","milvus","faiss","opensearch",
    "elasticsearch","vector database","vector db","vector store",
    "hybrid search","approximate nearest neighbor","hnsw",
    "chroma","chromadb","vespa","typesense","pgvector",
    "ndcg","mrr","mean reciprocal rank","map score","mean average precision",
    "learning to rank","lambdamart","ranknet","ranklib",
    "information retrieval","reranking","reranker","ranking system",
    "bm25","tf-idf","tfidf","sparse retrieval","two-stage retrieval",
    "recall@","precision@",
    "nlp","natural language processing","transformers","bert","roberta",
    "llm","large language model","rag","retrieval augmented",
    "text classification","named entity","question answering",
    "gpt","gemini","mistral","llama","langchain","llamaindex",
    "mlflow","mlops","model serving","triton","torchserve","bentoml",
    "feature store","online serving","ab testing","a/b testing",
    "shadow deployment","canary","drift detection",
    "python","fastapi","flask","celery","redis","kafka",
    "spark","airflow","dbt","sql","postgresql",
}
NICE_SKILLS = {
    "lora","qlora","peft","fine-tuning","fine tuning","sft","rlhf",
    "xgboost","lightgbm","catboost","gradient boosting",
    "distributed training","dask","horovod",
    "recommendation system","recommender system",
    "search relevance","query understanding","query expansion",
    "pytorch","tensorflow","jax","knowledge graph","neo4j","graph neural",
}
DISQ_TOKENS = {
    "computer vision","image classification","object detection",
    "image segmentation","yolo","resnet","vgg","convolutional",
    "speech recognition","asr","text-to-speech","tts","whisper",
    "robotics","ros framework","autonomous driving","slam",
}
CONSULTING_FIRMS = {
    "tata consultancy","tcs","infosys","wipro","accenture",
    "cognizant","capgemini","tech mahindra","hcl technologies",
    "mphasis","hexaware","l&t infotech","ltimindtree",
    "persistent systems","zensar","niit technologies",
    "mindtree","mastech","syntel","kpit",
}
STRONG_TITLES = {
    "ml engineer","machine learning engineer","ai engineer",
    "applied scientist","applied ml","research engineer",
    "nlp engineer","search engineer","ranking engineer",
    "recommendation","retrieval","data scientist",
    "senior software engineer","senior engineer","staff engineer",
    "principal engineer","founding engineer",
}
WRONG_TITLES = {
    "marketing manager","hr manager","human resources",
    "content writer","seo specialist","graphic designer",
    "ux designer","ui designer","sales executive",
    "project manager","program manager","scrum master",
    "finance manager","accountant","operations manager",
    "business development","product marketing","recruiter",
    "talent acquisition",
}
PREFERRED_LOCS = {
    "pune","noida","hyderabad","bengaluru","bangalore",
    "mumbai","delhi","gurgaon","gurugram","chennai",
    "kolkata","ahmedabad","ncr",
}

# ── Config ────────────────────────────────────────────────────────
WEIGHTS = {"skills":0.32,"career":0.28,"experience":0.18,"behavioral":0.14,"location":0.08}

def load_config(path="config.yaml"):
    try:
        content = Path(path).read_text()
        weights, in_w = {}, False
        for line in content.splitlines():
            if "scoring_weights:" in line: in_w=True; continue
            if in_w:
                m = re.match(r"\s+(\w+):\s*([\d.]+)", line)
                if m: weights[m.group(1)] = float(m.group(2))
                elif line and not line.startswith(" "): break
        if weights: WEIGHTS.update(weights); log.info(f"Config: {weights}")
    except: pass

# ── Helpers ───────────────────────────────────────────────────────
def days_ago(s):
    try: return (TODAY - date.fromisoformat(s)).days
    except: return 9999

def _compile(tokens):
    toks = sorted(tokens, key=len, reverse=True)
    return re.compile(r"\b("+"|".join(re.escape(t) for t in toks)+r")\b", re.IGNORECASE)

_CACHE = {}
def text_contains(text, tokens):
    key = id(tokens)
    if key not in _CACHE: _CACHE[key] = _compile(tokens)
    return len(set(_CACHE[key].findall(text.lower())))

def build_text(c):
    p = c.get("profile",{})
    parts = [p.get("summary",""), p.get("headline","")]
    for j in c.get("career_history",[]):
        parts += [j.get("title",""), j.get("description",""), j.get("company","")]
    for s in c.get("skills",[]): parts.append(s.get("name",""))
    for cert in c.get("certifications",[]): parts += [cert.get("name",""), cert.get("issuer","")]
    return " ".join(parts).lower()

def build_sem_text(c):
    """Richer text for semantic embedding — focuses on career descriptions."""
    p = c.get("profile",{})
    parts = [p.get("current_title",""), p.get("summary","")]
    for j in c.get("career_history",[])[:3]:
        parts += [j.get("title",""), j.get("description","")]
    return " ".join(x for x in parts if x)[:512]

# ── Honeypot ──────────────────────────────────────────────────────
def is_honeypot(c):
    skills = c.get("skills",[])
    career = c.get("career_history",[])
    yoe    = c.get("profile",{}).get("years_of_experience",0)
    if sum(1 for s in skills if s.get("proficiency") in ("expert","advanced")
           and s.get("duration_months",1)==0) >= 5: return True
    if sum(1 for s in skills if s.get("proficiency")=="expert"
           and 0 < s.get("duration_months",99) < 3) >= 3: return True
    total = sum(j.get("duration_months",0) for j in career)
    if yoe > 2 and total < yoe*12*0.4: return True
    return False

# ── Stage 1 Pillars ───────────────────────────────────────────────
def score_skills(c, txt):
    skills = c.get("skills",[])
    pw = {"expert":1.0,"advanced":0.75,"intermediate":0.45,"beginner":0.15}
    wc = sum(pw.get(s.get("proficiency","beginner"),0.15)*(1+(min(s.get("duration_months",0),72)/72))
             for s in skills if any(t in s["name"].lower() for t in CORE_SKILL_TOKENS))
    ws = min(wc/12,1); ts = min(text_contains(txt,CORE_SKILL_TOKENS)/8,1)
    ns = min(text_contains(txt,NICE_SKILLS)/4,1)*0.2
    dq = max(0,(text_contains(txt,DISQ_TOKENS)-text_contains(txt,{"nlp","natural language","retrieval","ranking","embedding"}))*0.08)
    return max(0,min(0.45*ws+0.40*ts+0.15*ns-dq,1))

def score_career(c, txt):
    profile = c.get("profile",{}); career = c.get("career_history",[])
    t = profile.get("current_title","").lower()
    if any(w in t for w in WRONG_TITLES): return 0.04
    ts = 0.25
    if any(s in t for s in STRONG_TITLES): ts = 0.75
    if any(e in t for e in ("ml engineer","ai engineer","nlp engineer","search engineer",
           "recommendation","retrieval","ranking engineer","applied scientist","staff engineer")): ts = 1.0
    tm=cm=0
    for j in career:
        d=j.get("duration_months",0); tm+=d
        if any(f in j.get("company","").lower() for f in CONSULTING_FIRMS): cm+=d
    cr = cm/tm if tm else 0
    ps = 0.05 if cr>=0.95 else 1.0-(cr*0.70)
    sigs={"shipped","deployed","production","serving","latency","a/b test","ab test",
          "experiment","offline eval","online eval","ndcg","mrr","precision@","recall@",
          "benchmark","embedding","retrieval","ranking","recommendation","vector","index",
          "rerank","feature store","pipeline"}
    dh = sum(sum(1 for s in sigs if s in j.get("description","").lower()) for j in career[:4])
    ds = min(dh/10,1)
    hp = min(sum(1 for j in career if j.get("duration_months",24)<12)*0.06,0.25)
    return max(0,min(0.35*ts+0.35*ps+0.30*ds-hp,1))

def score_exp(c):
    y=c.get("profile",{}).get("years_of_experience",0)
    if 6<=y<=8: return 1.00
    if 5<=y<6 or 8<y<=9: return 0.88
    if 4<=y<5 or 9<y<=11: return 0.65
    if 3<=y<4 or 11<y<=13: return 0.45
    if 13<y<=16: return 0.40
    if y>16: return 0.35
    return 0.10

def score_loc(c):
    p=c.get("profile",{}); sig=c.get("redrob_signals",{})
    loc=p.get("location","").lower(); country=p.get("country","").lower()
    rel=sig.get("willing_to_relocate",False)
    if country=="india":
        if any(x in loc for x in ("pune","noida","bengaluru","bangalore")): return 1.00
        if any(x in loc for x in PREFERRED_LOCS): return 0.90
        return 0.75 if rel else 0.60
    return 0.35 if rel else 0.15

def score_behav(c):
    sig=c.get("redrob_signals",{})
    d=days_ago(sig.get("last_active_date","2020-01-01"))
    rec=1.0 if d<=14 else 0.85 if d<=45 else 0.65 if d<=90 else 0.35 if d<=180 else 0.08
    otw=1.0 if sig.get("open_to_work_flag",False) else 0.40
    rr=sig.get("recruiter_response_rate",None); rr=float(rr) if rr is not None else 0.70
    n=sig.get("notice_period_days",90)
    ns=1.00 if n<=15 else 0.90 if n<=30 else 0.65 if n<=60 else 0.45 if n<=90 else 0.20
    ir=float(sig.get("interview_completion_rate",0.5))
    gh=sig.get("github_activity_score",-1); gs=gh/100 if gh>=0 else 0.25
    cp=sig.get("profile_completeness_score",50)/100
    return max(0,min(0.28*rec+0.22*otw+0.20*rr+0.15*ns+0.08*ir+0.04*gs+0.03*cp,1))

def rule_score(c):
    if is_honeypot(c): return 0.001, {"honeypot":True}
    txt=build_text(c)
    sk=score_skills(c,txt); ca=score_career(c,txt); ex=score_exp(c)
    be=score_behav(c); lo=score_loc(c)
    tot=WEIGHTS["skills"]*sk+WEIGHTS["career"]*ca+WEIGHTS["experience"]*ex+WEIGHTS["behavioral"]*be+WEIGHTS["location"]*lo
    return round(tot,6), {"skills":round(sk,3),"career":round(ca,3),"experience":round(ex,3),
                          "behavioral":round(be,3),"location":round(lo,3),"total":round(tot,4)}

def _worker(c):
    try:
        s,bd=rule_score(c); return (s,c["candidate_id"],bd)
    except Exception as e:
        return (0.0,c.get("candidate_id","UNK"),{"error":str(e)})

# ── Stage 2: Semantic ─────────────────────────────────────────────
def load_model():
    try:
        from sentence_transformers import SentenceTransformer
        path = str(MODEL_DIR) if MODEL_DIR.exists() else "sentence-transformers/all-MiniLM-L6-v2"
        log.info(f"Loading model from {path} ...")
        m = SentenceTransformer(path)
        log.info("Model loaded.")
        return m
    except ImportError:
        log.warning("sentence-transformers not installed. Run: pip install sentence-transformers && python download_model.py")
        return None
    except Exception as e:
        log.warning(f"Model load failed: {e} — falling back to rule-only")
        return None

def semantic_rerank(items, model, jd_text):
    import numpy as np
    log.info(f"Stage 2: Embedding {len(items)} candidates ...")
    t0=time.time()
    jd_emb=model.encode([jd_text],convert_to_numpy=True,show_progress_bar=False)
    jd_emb/=(np.linalg.norm(jd_emb,axis=1,keepdims=True)+1e-9)
    texts=[build_sem_text(x["candidate"]) for x in items]
    cemb=model.encode(texts,batch_size=64,convert_to_numpy=True,show_progress_bar=False)
    cemb/=(np.linalg.norm(cemb,axis=1,keepdims=True)+1e-9)
    sims=(jd_emb@cemb.T).flatten()
    for i,x in enumerate(items):
        ss=float(sims[i]); x["sem"]=round(ss,4)
        x["final"]=round(RULE_W*x["rule"]+SEM_W*ss,6)
    items.sort(key=lambda x:(-x["final"],x["candidate_id"]))
    log.info(f"Stage 2 done in {time.time()-t0:.1f}s")
    return items

# ── Reasoning ─────────────────────────────────────────────────────
def reasoning(c, bd, rank, sem=None):
    if bd.get("honeypot"): return "Flagged as invalid — honeypot pattern detected."
    p=c.get("profile",{}); sig=c.get("redrob_signals",{})
    pw={"expert":3,"advanced":2,"intermediate":1,"beginner":0}
    skills=c.get("skills",[])
    top_skills=sorted([s for s in skills if any(t in s["name"].lower() for t in CORE_SKILL_TOKENS)],
                      key=lambda s:pw.get(s.get("proficiency","beginner"),0),reverse=True)[:3]
    ss=", ".join(s["name"] for s in top_skills)
    title=p.get("current_title",""); yoe=p.get("years_of_experience",0); co=p.get("current_company","")
    d=days_ago(sig.get("last_active_date","2020-01-01"))
    rr=sig.get("recruiter_response_rate",0); n=sig.get("notice_period_days",90)
    otw=sig.get("open_to_work_flag",False)
    parts=[f"{title} {yoe:.1f}yrs @ {co}" + (f"; skills: {ss}" if ss else "") + "."]
    if sem is not None:
        lbl="strong" if sem>0.7 else "good" if sem>0.5 else "moderate"
        parts.append(f"JD semantic match: {lbl} ({sem:.2f}).")
    avail="actively available" if otw and d<=45 else f"inactive {d}d" if d>180 else f"active {d}d ago"
    parts.append(f"Signals: {avail}, resp {rr:.0%}, {n}d notice.")
    concerns=[]
    if bd.get("career",1)<0.30: concerns.append("limited product-company ML history")
    if bd.get("skills",1)<0.25: concerns.append("partial skill overlap")
    if p.get("country","").lower()!="india": concerns.append("non-India location")
    if concerns and rank>20: parts.append(f"Concern: {'; '.join(concerns)}.")
    return " ".join(parts)[:500]

# ── Main ──────────────────────────────────────────────────────────
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--candidates",default="candidates.jsonl")
    ap.add_argument("--out",default="submission.csv")
    ap.add_argument("--jd",default="jd.txt")
    ap.add_argument("--topn",type=int,default=100)
    ap.add_argument("--stage1-keep",type=int,default=STAGE1_KEEP)
    ap.add_argument("--config",default="config.yaml")
    ap.add_argument("--workers",type=int,default=max(1,mp.cpu_count()-1))
    ap.add_argument("--rules-only",action="store_true")
    args=ap.parse_args()

    load_config(args.config)
    log.info(f"Weights: {WEIGHTS}")

    jd_text=""
    try: jd_text=Path(args.jd).read_text(encoding="utf-8"); log.info(f"JD loaded ({len(jd_text)} chars)")
    except: log.warning("jd.txt not found")

    t0=time.time()
    log.info(f"Loading {args.candidates} ...")
    candidates=[]
    path=Path(args.candidates)
    opener=__import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path,"rt",encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if line:
                try: candidates.append(json.loads(line))
                except: pass
    log.info(f"Loaded {len(candidates):,} candidates")
    lookup={c["candidate_id"]:c for c in candidates}

    # Stage 1
    log.info("Stage 1: Rule scoring ...")
    t1=time.time()
    if len(candidates)>10_000 and args.workers>1:
        with mp.Pool(args.workers) as pool: results=pool.map(_worker,candidates)
    else: results=[_worker(c) for c in candidates]
    hp=sum(1 for r in results if r[2].get("honeypot"))
    log.info(f"Stage 1 done {time.time()-t1:.1f}s | honeypots={hp}")

    valid=[(s,cid,bd) for s,cid,bd in results if not bd.get("honeypot")]
    valid.sort(key=lambda x:(-x[0],x[1]))
    top_s1=valid[:args.stage1_keep]
    log.info(f"Stage 1 → kept {len(top_s1):,} for semantic reranking")

    # Stage 2
    use_sem = not args.rules_only and bool(jd_text)
    if use_sem:
        model=load_model(); use_sem = model is not None

    if use_sem:
        items=[{"candidate_id":cid,"rule":s,"bd":bd,"candidate":lookup[cid],"final":s,"sem":None}
               for s,cid,bd in top_s1]
        items=semantic_rerank(items,model,jd_text)
        top_n=items[:args.topn]
        log.info(f"Two-stage complete. Blend: rule={RULE_W}, semantic={SEM_W}")
    else:
        log.info("Rules-only mode")
        top_n=[{"candidate_id":cid,"rule":s,"bd":bd,"candidate":lookup[cid],"final":s,"sem":None}
               for s,cid,bd in top_s1[:args.topn]]

    # Write CSV
    out=Path(args.out)
    with open(out,"w",newline="",encoding="utf-8") as fh:
        w=csv.writer(fh); w.writerow(["candidate_id","rank","score","reasoning"])
        for rank,x in enumerate(top_n,1):
            w.writerow([x["candidate_id"],rank,f"{x['final']:.6f}",
                        reasoning(x["candidate"],x["bd"],rank,x.get("sem"))])

    log.info(f"Done → {out} | Total: {time.time()-t0:.1f}s")
    log.info("-- Top 10 --")
    for rank,x in enumerate(top_n[:10],1):
        p=x["candidate"]["profile"]
        sem=f"sem={x['sem']:.3f}" if x["sem"] else "sem=N/A"
        log.info(f"#{rank:>2} {x['candidate_id']} final={x['final']:.4f} rule={x['rule']:.4f} {sem} {p['current_title'][:25]} YOE={p['years_of_experience']:.1f} {p['location']}")

if __name__=="__main__": main()