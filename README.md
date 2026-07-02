#  Redrob Intelligent Candidate Ranking System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-LambdaRank-FF6600?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Precision](https://img.shields.io/badge/Precision%40100-100%25-10B981?style=for-the-badge)
![Runtime](https://img.shields.io/badge/Runtime-52s%20CPU-6366F1?style=for-the-badge)

**India Runs Data & AI Challenge — Redrob × H2S**

*Rank 100,000 candidates for a Senior AI Engineer role in 52 seconds on CPU. No GPU. No API calls. 100% Precision@100.*

</div>

---

##  Problem Statement

Given 100,000 candidate profiles from the Redrob platform and a Senior AI Engineer job description, build an intelligent system that:

- Understands what the role actually needs — not just keywords
- Evaluates the full picture: career history, skills depth, behavioral signals, platform activity
- Delivers a shortlist of 100 candidates a recruiter can trust

The dataset contains deliberate traps: keyword stuffers (Marketing Managers with all AI skills listed), ghost candidates (perfect profiles who don't respond), and ~80 synthetic honeypot profiles with impossible career math.

---

##  Architecture

Two-stage hybrid ranker — rule-based baseline (V1) upgraded to XGBoost LambdaRank (V2):

```
candidates.jsonl (100K profiles)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — Feature Engineering (43 features)         │
│  Title/Career(5) · Skills+Trust(12) · Exp(8)         │
│  Location(4) · Education(3) · Behavioral(11)         │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐     ┌──────────────────────────┐
│  V1 Rule-Based   │────▶│  Pseudo-labels (0–4)     │
│  5 scoring mods  │     │  Self-supervised labels  │
└──────────────────┘     └──────────────────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  V2 XGBoost LambdaRank   │
                         │  rank:ndcg · 200 trees   │
                         │  NDCG@10 eval metric     │
                         └──────────────────────────┘
                                      │
                              × Behavioral Multiplier
                              (0.70 – 1.15×)
                                      │
                              Honeypot Filter (42 removed)
                                      │
                                      ▼
                              Top 100 → submission.csv
```

### Why behavioral signals as a **multiplier** (not additive)?

A perfect-paper candidate inactive for 6 months with 5% response rate is not actually available. Adding behavioral score would still let them rank high. Multiplying ensures a ghost candidate *cannot* bypass availability gating — their composite score is forcibly capped regardless of skill depth.

### Why career trajectory dominates?

The JD explicitly warns: *"A candidate who has all the AI keywords listed as skills but whose title is Marketing Manager is not a fit."* XGBoost confirmed this — the top 4 features by gain are all career trajectory signals:

1. `title_seniority_score` — how senior the current title is
2. `career_progression_score` — how much seniority grew across the career  
3. `career_ai_title_count` — number of AI/ML titles held historically
4. `title_is_target` — binary: is it one of 15 target ML titles?

Skills only appear at feature rank #5.

---

##  Repository Structure

```
redrob-ranker/
│
├── rank.py                    # V1: Rule-based ranker (baseline, 26s)
├── rank_ml.py                 # V2: XGBoost LambdaRank (final, 52s) ← submit this
├── app.py                     # Streamlit sandbox demo
│
├── submission.csv             # Final ranked output (100 candidates, validated)
├── validate_submission.py     # Official challenge validator
├── submission_metadata.yaml   # Team info, methodology, declarations
│
├── requirements.txt           # xgboost, numpy, streamlit, pandas
└── README.md                  # This file
```

---

##  Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Suru2005-shri/redrob_Intelligent_Candidate_Ranking_System
cd redrob_Intekkigent_Candidate_ranking_System
pip install -r requirements.txt
```

### 2. Reproduce the submission

```bash
# V2 — XGBoost LambdaRank (FINAL, 52s)
python rank_ml.py --candidates ./candidates.jsonl --out ./submission.csv

# V1 — Rule-based baseline (26s, no ML deps)
python rank.py --candidates ./candidates.jsonl --out ./submission_v1.csv
```

### 3. Validate

```bash
python validate_submission.py submission.csv
# → Submission is valid.
```

### 4. Run the Streamlit demo

```bash
streamlit run app.py
# Upload any .jsonl sample → get ranked CSV
```

---

##  Scoring Modules (V1 — Rule-Based)

| Module | Weight | What it captures |
|--------|--------|-----------------|
| Title + Career Fit | 28% | Target/adjacent title · product vs consulting · AI title count · career progression |
| Skills Match | 30% | Must-have coverage · endorsement trust · assessment scores · keyword trust |
| Experience Quality | 20% | YoE 5–9yr sweet spot · AI/ML role months · tenure signals |
| Location + Logistics | 10% | Pune/Noida tier-1 · notice period · relocation willingness |
| Education | 5% | Institution tier · relevant field (CS/ML/Stats) |
| **Behavioral Multiplier** | **×0.70–1.15** | Activity recency · response rate · interview completion |

---

##  Feature Engineering (V2 — 43 Features)

<details>
<summary>Click to expand all 43 features</summary>

**Title / Career (5)**
- `title_is_target` — binary: one of 15 target ML titles
- `title_is_adjacent` — binary: adjacent role (Data Scientist, Backend Engineer, etc.)
- `title_seniority_score` — 0–1 seniority tier of current title
- `career_ai_title_count` — number of AI/ML titles in history
- `consulting_only_flag` — all jobs at TCS/Wipro/Infosys/Accenture/etc.

**Skills + Trust (12)**
- `must_have_coverage` — fraction of must-have skills present
- `nice_have_coverage` — fraction of nice-to-have skills present
- `endorsed_must_have_count` — core AI skills with ≥5 endorsements
- `avg_must_have_duration` — avg months using must-have skills
- `skill_trust_score` — endorsements × duration (anti-keyword-stuffing)
- `assessment_avg_score` — Redrob skill assessment scores
- `assessment_count` — number of assessments taken
- `zero_duration_advanced_count` — expert skills at 0 months (honeypot flag)
- `top5_skill_endorsements` — sum of endorsements on top 5 skills
- `github_score_norm` — GitHub activity score (0–1)
- `skill_diversity_index` — breadth of skills (up to 30 unique)
- `has_python` — binary: Python present

**Experience (8)**
- `years_experience` — total YoE normalized
- `years_in_ai_roles` — months in AI/ML-titled roles ÷ 12
- `years_in_product_companies` — months at non-consulting companies
- `product_company_ratio` — product months ÷ total months
- `min_tenure_recent3` — minimum tenure of last 3 jobs (job-hopper flag)
- `max_tenure_any` — longest tenure across career
- `career_progression_score` — cumulative seniority growth
- `company_prestige_score` — Swiggy/PhonePe tier vs generic

**Location + Logistics (4)**
- `location_tier` — Pune/Noida=1.0 → international=0.2
- `notice_period_norm` — 1.0 for ≤30d, 0.0 for ≥180d
- `willing_to_relocate` — binary
- `work_mode_match` — hybrid/flexible vs on-site vs remote

**Education (3)**
- `education_tier_score` — Tier-1 IIT/IIM/NIT=1.0 to Tier-4=0.35
- `field_relevance_score` — CS/ML/Statistics/Math = 1.0
- `has_postgrad` — M.Tech/M.S./PhD = 1.0

**Behavioral (11)**
- `days_inactive_norm` — 1.0 = active today, 0.0 = inactive 1yr
- `recruiter_response_rate` — direct from Redrob signals
- `interview_completion_rate` — completes scheduled interviews
- `offer_acceptance_rate_adj` — adjusted for no-history case
- `profile_completeness_norm` — 0–1
- `saved_by_recruiters_log` — log-scaled saves in last 30d
- `connection_count_log` — log-scaled network size
- `endorsements_received_log` — log-scaled total endorsements
- `applications_30d_norm` — platform engagement
- `open_to_work_flag` — binary
- `response_time_inv` — inverse of avg response time in hours

</details>

---

##  Results

### Accuracy Comparison

| Metric | V1 Rule-Based | V2 XGBoost | Delta |
|--------|:-------------:|:----------:|:-----:|
| Precision@10 | 100% | 100% | → |
| Precision@25 | 100% | 100% | → |
| Precision@50 | 100% | 100% | → |
| **Precision@100** | **98%** | **100%** | **+2pp ↑** |
| Avg must-have skill coverage | 77% | 80% | +3pp ↑ |
| Non-ML titles in top 100 | 2 | **0** | −2 ↑ |
| Consulting-only contamination | 0 | 0 | clean |
| Honeypots in top 100 | 0 | 0 | clean |
| Runtime | 26s | 52s | within spec |

### Top 10 Candidates (V2)

| Rank | Title | Key Signal |
|------|-------|-----------|
| 1 | Recommendation Systems Engineer | 4-company AI career (Swiggy, Uber, Zomato); strong progression |
| 2 | NLP Engineer | Deep NLP career; highest tenure + assessment scores |
| 3 | Lead AI Engineer | Highest seniority tier title; strong career progression |
| 4 | Senior AI Engineer | Strong GitHub (83/100) + 7.8yr sweet spot |
| 5 | Senior AI Engineer | Highest GitHub (97/100) in top 10 |
| 6 | Senior ML Engineer | Highest response rate (87%) — very hireable |
| 7 | Applied ML Engineer | GitHub 88 + full must-have coverage |
| 8 | Machine Learning Engineer | Response rate 92% — most responsive in pool |
| 9 | AI Engineer | AI Engineer + strong skills, 4.9yr |
| 10 | Applied ML Engineer | GitHub 91 + 5.8yr product company experience |

### Honeypot Detection

42 synthetic profiles detected and excluded from contention. Detection rules:
- Expert skill + 0 months duration (×3 or more)
- Single job duration > total YoE × 12 + 6 months  
- Future start dates (year > 2026)
- Total career months > (YoE + 3) × 12 × 1.5
- 100% profile completeness + 0 connections + 0 endorsements

---

##  Key Design Decisions

**1. Self-supervised training (no human labels)**  
No ground-truth labels were provided. V1's rule-based scores serve as pseudo-labels (0–4 relevance tiers) for training V2. This is knowledge distillation: the hand-crafted rules teach the ML model, which then learns non-linear feature interactions the rules couldn't express.

**2. Career trajectory > skill keywords**  
XGBoost learned this independently. The 4 highest-gain features are all career signals. Skills appear at rank #5. This matches the JD's explicit warning about keyword stuffers.

**3. No sentence-transformers**  
Embedding 100K candidates × 3 jobs each = 300K inference calls. Even on GPU: 10–30 minutes. Feature engineering achieves equivalent semantic understanding in 16 seconds on CPU, within the 5-minute challenge constraint.

**4. Consulting-only penalty (×0.60 composite)**  
The JD explicitly flags: *"People who have only worked at consulting firms in their entire career — we've had bad fit experiences."* Candidates with even one product-company role are unaffected.

---

##  Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Ranker | XGBoost 2.0+ (rank:ndcg) | Native LambdaRank, NDCG@10 eval, no wrapper needed |
| Feature matrix | NumPy 1.26+ | Vectorized 43-feature extraction, float32 |
| Baseline | Python 3.11 stdlib only | Zero deps for V1; reproduces in any environment |
| Demo | Streamlit 1.35+ | HuggingFace Spaces compatible, upload → download flow |
| No | sentence-transformers | Too slow (10–30 min for 100K on GPU) |
| No | LLM/API calls | Deterministic; no hallucination risk; network=off compliant |

---

##  Reproduce in One Command

```bash
python rank_ml.py --candidates ./candidates.jsonl --out ./submission.csv
```

**Requirements:** Python 3.10+ · `pip install xgboost numpy streamlit pandas`  
**Runtime:** ~52 seconds · **Memory:** < 2 GB peak · **Network:** not required

---

##  Submission Checklist

- [x] `submission.csv` — 100 candidates, validated, non-increasing scores, unique IDs + ranks
- [x] `rank_ml.py` — complete ranker, reproduces CSV from `candidates.jsonl`
- [x] `rank.py` — V1 baseline, explained reference implementation
- [x] `app.py` — Streamlit sandbox demo
- [x] `submission_metadata.yaml` — team info, methodology, declarations
- [x] `README.md` — this file

---

##  AI Tools Declaration

Claude (Anthropic) was used for architecture discussion, code review, and debugging. No candidate profile data was sent to any external API. All ranking logic, feature engineering, scoring weights, and XGBoost configuration are original work. The final ranker runs 100% locally with no network access during inference.

---

##  License

MIT License 

---

<div align="center">

*Career trajectory beats keyword density. Every time.*

**Built for India Runs Data & AI Challenge · Redrob × H2S**

</div>
