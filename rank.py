#!/usr/bin/env python3
"""
Redrob Intelligent Candidate Ranking System
============================================
A multi-signal hybrid ranker for the India Runs Data & AI Challenge.

Architecture:
  1. Fast pre-filter: coarse elimination of clearly irrelevant candidates
  2. Feature extraction: 5 scoring modules over profile + behavioral signals
  3. Composite scoring: weighted combination with behavioral multiplier
  4. Honeypot detection: catches impossible profiles automatically
  5. Reasoning generation: human-readable per-candidate justification

Runtime: < 5 min on 16 GB CPU for 100K candidates.
No API calls. No GPU. No network during ranking.
"""

import json
import math
import re
import csv
import argparse
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ROLE KNOWLEDGE — extracted from JD analysis
# ─────────────────────────────────────────────────────────────────────────────

# Hard must-have signals from JD
MUST_HAVE_SKILLS = {
    # embeddings / retrieval
    "sentence-transformers", "sentence transformers", "openai embeddings",
    "bge", "e5", "embeddings", "vector search", "dense retrieval",
    "hybrid retrieval", "hybrid search", "semantic search",
    # vector DBs
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "vector db",
    # eval / ranking
    "ndcg", "mrr", "map", "learning to rank", "learning-to-rank",
    "ranking", "information retrieval",
    # core
    "python", "nlp", "information retrieval", "retrieval",
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine-tune", "fine tuning",
    "xgboost", "lightgbm", "recommendation", "recommender",
    "a/b testing", "a/b test", "ab test", "distributed systems",
    "kafka", "spark", "airflow", "rag", "llm", "transformer",
    "bert", "pytorch", "tensorflow", "hugging face", "huggingface",
    "bm25", "elasticsearch", "solr",
}

# Titles that are strongly aligned
TARGET_TITLES = {
    "senior ai engineer", "lead ai engineer", "ai engineer",
    "applied ml engineer", "senior applied scientist", "applied scientist",
    "staff machine learning engineer", "senior machine learning engineer",
    "machine learning engineer", "ml engineer", "nlp engineer",
    "senior nlp engineer", "search engineer", "recommendation systems engineer",
    "senior software engineer (ml)", "senior data scientist",
}

# Titles that are adjacent / partial fit
ADJACENT_TITLES = {
    "data scientist", "analytics engineer", "backend engineer",
    "software engineer", "senior software engineer",
    "full stack developer", "data engineer", "senior data engineer",
    "computer vision engineer", "ai research engineer",
    "ai specialist", "junior ml engineer",
}

# Explicit disqualifiers from JD
DISQUALIFIED_COMPANIES = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "tata consultancy", "hcl", "tech mahindra",
}

# Core AI skill vocabulary for counting
AI_CORE_SKILLS = {
    "python", "machine learning", "deep learning", "nlp",
    "natural language processing", "pytorch", "tensorflow",
    "transformers", "bert", "gpt", "llm", "embeddings",
    "vector search", "faiss", "elasticsearch", "qdrant", "pinecone",
    "milvus", "weaviate", "sentence-transformers", "bge", "e5",
    "hybrid retrieval", "dense retrieval", "bm25", "information retrieval",
    "ranking", "learning to rank", "ndcg", "a/b testing",
    "rag", "retrieval", "recommendation", "lora", "qlora", "fine-tuning",
}

# Consulting-only penalizer
PRODUCT_COMPANY_INDICATORS = {
    "startup", "saas", "product", "internet", "fintech", "edtech",
    "healthtech", "ai/ml", "consumer", "marketplace", "ecommerce",
}

REFERENCE_DATE = date(2026, 6, 28)


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def is_honeypot(c: dict) -> bool:
    """Detect candidates with subtly impossible profiles."""
    flags = 0

    profile = c.get("profile", {})
    career = c.get("career_history", [])
    skills = c.get("skills", [])

    yoe = profile.get("years_of_experience", 0)

    # Flag 1: Company founded after years_of_experience would allow
    for job in career:
        start = job.get("start_date", "")
        if start:
            try:
                start_year = int(start[:4])
                # If they claim to have started before they could have (< 18yo + edu time)
                # We can't know birth year but implausible if start_year + yoe > 2026+5
                if start_year > 2026:
                    flags += 2
            except Exception:
                pass
        dur = job.get("duration_months", 0)
        if dur > yoe * 12 + 6:  # single job longer than total career
            flags += 2

    # Flag 2: Expert in skill with 0 months duration
    expert_zero = sum(
        1 for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero >= 3:
        flags += 2

    # Flag 3: Too many expert skills total with 0 duration
    zero_dur_advanced = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) == 0
    )
    if zero_dur_advanced >= 5:
        flags += 1

    # Flag 4: Total career duration > years_of_experience * 1.5 (overlapping jobs w/o reason)
    total_months = sum(j.get("duration_months", 0) for j in career)
    if total_months > (yoe + 3) * 12 * 1.5:
        flags += 1

    # Flag 5: Profile completeness 100 but 0 connections and 0 endorsements
    rs = c.get("redrob_signals", {})
    if (rs.get("profile_completeness_score", 0) == 100
            and rs.get("connection_count", 0) == 0
            and rs.get("endorsements_received", 0) == 0):
        flags += 2

    return flags >= 2


# ─────────────────────────────────────────────────────────────────────────────
# SCORING MODULES
# ─────────────────────────────────────────────────────────────────────────────

def score_title_and_career(c: dict) -> tuple[float, list[str]]:
    """
    Title + career trajectory score (0-1).
    This is the decisive signal against keyword stuffers.
    """
    notes = []
    score = 0.0

    profile = c.get("profile", {})
    career = c.get("career_history", [])

    current_title = profile.get("current_title", "").lower()

    # Current title match
    if current_title in TARGET_TITLES:
        score += 0.40
        notes.append(f"target title: {profile['current_title']}")
    elif current_title in ADJACENT_TITLES:
        score += 0.20
        notes.append(f"adjacent title: {profile['current_title']}")
    else:
        notes.append(f"non-target title: {profile['current_title']}")

    # Career history: product company presence
    has_product_company = False
    all_consulting = True
    product_titles_in_history = 0

    for job in career:
        company = job.get("company", "").lower()
        industry = job.get("industry", "").lower()
        title = job.get("title", "").lower()
        is_consulting = any(dc in company for dc in DISQUALIFIED_COMPANIES)
        is_product = any(pi in industry for pi in PRODUCT_COMPANY_INDICATORS)

        if not is_consulting:
            all_consulting = False
        if is_product or not is_consulting:
            has_product_company = True

        if any(t in title for t in ["ml", "ai", "search", "ranking", "nlp", "data sci", "recommendation"]):
            product_titles_in_history += 1

    if has_product_company:
        score += 0.25
        notes.append("product company background")
    if all_consulting:
        score -= 0.30
        notes.append("WARNING: all consulting companies")

    if product_titles_in_history >= 2:
        score += 0.20
        notes.append(f"{product_titles_in_history} ML/AI roles in history")
    elif product_titles_in_history == 1:
        score += 0.10

    # Check for explicit disqualifiers mentioned in JD
    current_company = profile.get("current_company", "").lower()
    all_current_consulting = any(dc in current_company for dc in DISQUALIFIED_COMPANIES)

    has_prior_product = False
    for job in career:
        if not job.get("is_current") and not any(
            dc in job.get("company", "").lower() for dc in DISQUALIFIED_COMPANIES
        ):
            has_prior_product = True

    if all_current_consulting and not has_prior_product:
        score -= 0.20
        notes.append("only consulting experience")

    return max(0.0, min(1.0, score)), notes


def score_skills(c: dict) -> tuple[float, list[str]]:
    """
    Skills match score (0-1).
    Weighted: must-haves > nice-to-haves, with endorsement+duration trust.
    """
    notes = []
    skills = c.get("skills", [])

    skill_names_lower = {s["name"].lower() for s in skills}
    skill_map = {s["name"].lower(): s for s in skills}

    # Must-have coverage
    must_have_matches = MUST_HAVE_SKILLS & skill_names_lower
    # Also check substrings for compound skill names
    for mh in MUST_HAVE_SKILLS:
        for sn in skill_names_lower:
            if mh in sn or sn in mh:
                must_have_matches.add(mh)

    nice_to_have_matches = NICE_TO_HAVE_SKILLS & skill_names_lower
    for nth in NICE_TO_HAVE_SKILLS:
        for sn in skill_names_lower:
            if nth in sn or sn in nth:
                nice_to_have_matches.add(nth)

    # Base coverage score
    must_ratio = min(len(must_have_matches) / 6, 1.0)  # cap at 6 must-haves
    nice_ratio = min(len(nice_to_have_matches) / 5, 1.0)

    base_score = 0.60 * must_ratio + 0.25 * nice_ratio

    # Trust multiplier: endorsements + duration (anti-keyword-stuffing)
    trust_bonus = 0.0
    for skill in skills:
        sname = skill["name"].lower()
        if any(mh in sname or sname in mh for mh in MUST_HAVE_SKILLS):
            endorsements = skill.get("endorsements", 0)
            duration = skill.get("duration_months", 0)
            proficiency = skill.get("proficiency", "beginner")

            # Penalize zero-duration claimed advanced skills (keyword stuffing signal)
            if proficiency in ("advanced", "expert") and duration < 3:
                trust_bonus -= 0.02
            elif duration >= 12 and endorsements >= 5:
                trust_bonus += 0.02

    base_score = max(0.0, min(1.0, base_score + trust_bonus))

    # Skill assessment scores bonus
    assessments = c.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        base_score += 0.15 * (avg_assessment / 100)

    notes.append(f"{len(must_have_matches)} must-have skills matched")
    if nice_to_have_matches:
        notes.append(f"{len(nice_to_have_matches)} nice-to-have skills")
    if assessments:
        notes.append(f"assessments avg: {sum(assessments.values())/len(assessments):.0f}")

    return max(0.0, min(1.0, base_score)), notes


def score_experience(c: dict) -> tuple[float, list[str]]:
    """
    Experience quality score (0-1).
    Considers YoE, relevance of YoE (AI-specific roles), career velocity.
    """
    notes = []
    profile = c.get("profile", {})
    career = c.get("career_history", [])

    yoe = profile.get("years_of_experience", 0)

    # JD wants 5-9 years (sweet spot), open to 4-12 with caveats
    if 5 <= yoe <= 9:
        yoe_score = 1.0
        notes.append(f"{yoe:.1f} yrs (sweet spot 5-9)")
    elif 4 <= yoe < 5 or 9 < yoe <= 12:
        yoe_score = 0.75
        notes.append(f"{yoe:.1f} yrs (slightly outside 5-9)")
    elif yoe < 4:
        yoe_score = 0.40
        notes.append(f"{yoe:.1f} yrs (too junior)")
    else:  # > 12
        yoe_score = 0.60
        notes.append(f"{yoe:.1f} yrs (overqualified risk)")

    # AI-specific experience months
    ai_months = 0
    for job in career:
        title = job.get("title", "").lower()
        desc = job.get("description", "").lower()
        duration = job.get("duration_months", 0)

        is_ai_role = any(kw in title for kw in [
            "ml", "ai", "machine learning", "data sci", "nlp",
            "search", "ranking", "recommendation", "applied scientist"
        ])
        has_ai_desc = sum(1 for kw in [
            "embedding", "retrieval", "ranking", "nlp", "vector",
            "model", "recommendation", "search"
        ] if kw in desc) >= 3

        if is_ai_role or has_ai_desc:
            ai_months += duration

    ai_yoe = ai_months / 12
    if ai_yoe >= 4:
        ai_bonus = 0.25
        notes.append(f"{ai_yoe:.1f} yrs in AI/ML roles")
    elif ai_yoe >= 2:
        ai_bonus = 0.15
        notes.append(f"{ai_yoe:.1f} yrs in AI-adjacent roles")
    else:
        ai_bonus = 0.0

    # Tenure signals: JD wants 3+ year stayers, not 1.5yr hoppers
    recent_jobs = sorted(career, key=lambda x: x.get("start_date", ""), reverse=True)[:3]
    short_tenures = sum(1 for j in recent_jobs if j.get("duration_months", 24) < 18)
    if short_tenures >= 2:
        tenure_penalty = -0.10
        notes.append("frequent job hopping detected")
    elif any(j.get("duration_months", 0) >= 36 for j in career[:2]):
        tenure_penalty = 0.05
        notes.append("good tenure signals")
    else:
        tenure_penalty = 0.0

    score = 0.60 * yoe_score + ai_bonus + tenure_penalty
    return max(0.0, min(1.0, score)), notes


def score_location_fit(c: dict) -> tuple[float, list[str]]:
    """
    Location and logistics fit (0-1).
    JD: Pune/Noida preferred, open to Hyderabad/Mumbai/Delhi NCR.
    """
    notes = []
    profile = c.get("profile", {})
    rs = c.get("redrob_signals", {})

    country = profile.get("country", "").lower()
    location = profile.get("location", "").lower()
    willing_to_relocate = rs.get("willing_to_relocate", False)
    notice_period = rs.get("notice_period_days", 90)

    # Location tiers
    tier1_locations = ["noida", "pune", "new delhi", "delhi"]
    tier2_locations = ["hyderabad", "mumbai", "bangalore", "bengaluru", "gurgaon", "gurugram", "ncr"]
    tier3_india = country == "india"

    if any(loc in location for loc in tier1_locations):
        loc_score = 1.0
        notes.append(f"preferred location: {profile.get('location')}")
    elif any(loc in location for loc in tier2_locations):
        loc_score = 0.85
        notes.append(f"tier-2 preferred city: {profile.get('location')}")
    elif tier3_india:
        if willing_to_relocate:
            loc_score = 0.70
            notes.append(f"India-based, willing to relocate")
        else:
            loc_score = 0.55
            notes.append(f"India-based, not willing to relocate")
    else:
        if willing_to_relocate:
            loc_score = 0.35
            notes.append("international, willing to relocate")
        else:
            loc_score = 0.15
            notes.append("international, not relocating")

    # Notice period score
    if notice_period <= 30:
        notice_score = 1.0
        notes.append(f"notice: {notice_period}d (ideal)")
    elif notice_period <= 60:
        notice_score = 0.75
        notes.append(f"notice: {notice_period}d (acceptable)")
    elif notice_period <= 90:
        notice_score = 0.50
        notes.append(f"notice: {notice_period}d (long)")
    else:
        notice_score = 0.25
        notes.append(f"notice: {notice_period}d (very long)")

    score = 0.70 * loc_score + 0.30 * notice_score
    return max(0.0, min(1.0, score)), notes


def score_education(c: dict) -> tuple[float, list[str]]:
    """
    Education signal (0-1). Lower weight - JD doesn't emphasize it strongly.
    """
    notes = []
    education = c.get("education", [])

    if not education:
        return 0.30, ["no education data"]

    best_tier = "tier_4"
    best_field_match = False
    has_postgrad = False

    tier_order = {"tier_1": 0, "tier_2": 1, "tier_3": 2, "tier_4": 3, "unknown": 4}
    relevant_fields = {"computer science", "information technology", "ai", "machine learning",
                       "data science", "statistics", "mathematics", "electrical engineering",
                       "electronics", "software"}

    for edu in education:
        tier = edu.get("tier", "unknown")
        if tier_order.get(tier, 4) < tier_order.get(best_tier, 4):
            best_tier = tier

        field = edu.get("field_of_study", "").lower()
        if any(rf in field for rf in relevant_fields):
            best_field_match = True

        degree = edu.get("degree", "").lower()
        if any(pg in degree for pg in ["m.tech", "m.e.", "m.s.", "msc", "phd", "m.b.a", "master"]):
            has_postgrad = True

    tier_scores = {"tier_1": 1.0, "tier_2": 0.80, "tier_3": 0.55, "tier_4": 0.35, "unknown": 0.30}
    tier_score = tier_scores.get(best_tier, 0.30)

    field_bonus = 0.15 if best_field_match else 0.0
    postgrad_bonus = 0.10 if has_postgrad else 0.0

    notes.append(f"education tier: {best_tier}")
    if best_field_match:
        notes.append("relevant field of study")

    score = tier_score + field_bonus + postgrad_bonus
    return max(0.0, min(1.0, score)), notes


def score_behavioral_signals(c: dict) -> tuple[float, list[str]]:
    """
    Behavioral engagement score (0-1).
    Acts as a MULTIPLIER on top of profile quality — not an additive component.
    A great-profile but ghost candidate should be heavily down-weighted.
    """
    notes = []
    rs = c.get("redrob_signals", {})

    # Recency: last active date
    last_active_str = rs.get("last_active_date", "2020-01-01")
    try:
        last_active = date.fromisoformat(last_active_str)
        days_inactive = (REFERENCE_DATE - last_active).days
        if days_inactive <= 30:
            activity_score = 1.0
            notes.append("active in last 30 days")
        elif days_inactive <= 90:
            activity_score = 0.80
            notes.append(f"active {days_inactive}d ago")
        elif days_inactive <= 180:
            activity_score = 0.55
            notes.append(f"inactive {days_inactive}d")
        else:
            activity_score = 0.20
            notes.append(f"INACTIVE {days_inactive}d — availability concern")
    except Exception:
        activity_score = 0.50

    # Recruiter response rate (key signal for hireability)
    response_rate = rs.get("recruiter_response_rate", 0.0)
    if response_rate >= 0.70:
        response_score = 1.0
        notes.append(f"response rate: {response_rate:.0%} (high)")
    elif response_rate >= 0.40:
        response_score = 0.75
        notes.append(f"response rate: {response_rate:.0%}")
    elif response_rate >= 0.20:
        response_score = 0.45
        notes.append(f"response rate: {response_rate:.0%} (low)")
    else:
        response_score = 0.15
        notes.append(f"response rate: {response_rate:.0%} (very low)")

    # Open to work
    otw_bonus = 0.10 if rs.get("open_to_work_flag") else 0.0

    # Interview completion rate
    icr = rs.get("interview_completion_rate", 0.5)
    icr_score = icr  # 0-1, direct mapping

    # Profile completeness
    completeness = rs.get("profile_completeness_score", 50) / 100
    completeness_score = 0.50 + 0.50 * completeness  # 0.5 to 1.0

    # GitHub activity (bonus for AI roles)
    github = rs.get("github_activity_score", -1)
    github_bonus = 0.0
    if github >= 50:
        github_bonus = 0.10
        notes.append(f"strong GitHub: {github:.0f}")
    elif github >= 20:
        github_bonus = 0.05

    # Saved by recruiters (wisdom of crowds)
    saved = rs.get("saved_by_recruiters_30d", 0)
    saved_bonus = min(saved / 20, 0.10)

    # Composite behavioral score
    score = (
        0.35 * activity_score
        + 0.30 * response_score
        + 0.15 * icr_score
        + 0.10 * completeness_score
        + otw_bonus
        + github_bonus
        + saved_bonus
    )

    return max(0.0, min(1.0, score)), notes


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORER
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "title_career":  0.28,
    "skills":        0.30,
    "experience":    0.20,
    "location":      0.10,
    "education":     0.05,
}


def score_candidate(c: dict) -> dict:
    """Full scoring pipeline for one candidate."""

    if is_honeypot(c):
        return {
            "candidate_id": c["candidate_id"],
            "composite": 0.0,
            "behavioral": 0.0,
            "notes": ["HONEYPOT: impossible profile detected"],
            "is_honeypot": True,
        }

    tc_score, tc_notes = score_title_and_career(c)
    sk_score, sk_notes = score_skills(c)
    ex_score, ex_notes = score_experience(c)
    lo_score, lo_notes = score_location_fit(c)
    ed_score, ed_notes = score_education(c)
    bh_score, bh_notes = score_behavioral_signals(c)

    # Profile quality (static signals)
    profile_score = (
        WEIGHTS["title_career"] * tc_score
        + WEIGHTS["skills"] * sk_score
        + WEIGHTS["experience"] * ex_score
        + WEIGHTS["location"] * lo_score
        + WEIGHTS["education"] * ed_score
    )

    # Behavioral multiplier: 0.7 to 1.15
    # A ghost candidate (bh_score ~0.2) gets 0.74x, a highly engaged one gets 1.15x
    behavioral_multiplier = 0.70 + 0.45 * bh_score

    composite = profile_score * behavioral_multiplier

    # Explicit disqualifier check: consulting-only
    profile = c.get("profile", {})
    career = c.get("career_history", [])
    current_company = profile.get("current_company", "").lower()
    all_consulting = all(
        any(dc in job.get("company", "").lower() for dc in DISQUALIFIED_COMPANIES)
        for job in career
    )
    if all_consulting and len(career) >= 2:
        composite *= 0.60

    all_notes = tc_notes + sk_notes + ex_notes + lo_notes + bh_notes

    return {
        "candidate_id": c["candidate_id"],
        "composite": composite,
        "behavioral": bh_score,
        "tc_score": tc_score,
        "sk_score": sk_score,
        "ex_score": ex_score,
        "lo_score": lo_score,
        "ed_score": ed_score,
        "notes": all_notes,
        "is_honeypot": False,
        "profile": c.get("profile", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# REASONING GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(result: dict, rank: int) -> str:
    """
    Generate specific, honest 1-2 sentence reasoning for each candidate.
    Avoids hallucination by only referencing facts from the result dict.
    """
    if result.get("is_honeypot"):
        return "Profile contains inconsistencies suggesting synthetic/honeypot data; excluded."

    profile = result.get("profile", {})
    notes = result.get("notes", [])
    title = profile.get("current_title", "unknown")
    yoe = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    location = profile.get("location", "")
    country = profile.get("country", "")

    # Pick the most informative notes
    skill_note = next((n for n in notes if "skill" in n.lower() or "must-have" in n.lower()), "")
    title_note = next((n for n in notes if "title" in n.lower()), "")
    loc_note = next((n for n in notes if "location" in n.lower() or "notice" in n.lower() or "active" in n.lower()), "")
    activity_note = next((n for n in notes if "active" in n.lower() or "response" in n.lower()), "")
    ai_exp_note = next((n for n in notes if "ai/ml" in n.lower() or "yrs in" in n.lower()), "")

    parts = []

    # Core qualification sentence
    if result.get("tc_score", 0) >= 0.60:
        parts.append(f"{title} at {company} with {yoe:.1f} yrs; strong title-career match for senior AI engineer role")
    elif result.get("tc_score", 0) >= 0.35:
        parts.append(f"{title} with {yoe:.1f} yrs exp; adjacent background with relevant transferable signals")
    else:
        parts.append(f"{title} ({yoe:.1f} yrs); weak title alignment but included based on other signals")

    if skill_note:
        parts.append(skill_note)
    if ai_exp_note:
        parts.append(ai_exp_note)

    # Engagement / availability sentence
    concerns = []
    if activity_note and "inactive" in activity_note.lower():
        concerns.append(activity_note.lower())
    if "response rate" in " ".join(notes) and "low" in " ".join(notes):
        concerns.append("low recruiter response rate")
    if "all consulting" in " ".join(notes).lower():
        concerns.append("consulting-only background")

    if concerns:
        parts.append("Concern: " + "; ".join(concerns))
    elif loc_note:
        parts.append(loc_note)

    reasoning = ". ".join(parts[:3])
    if len(reasoning) > 250:
        reasoning = reasoning[:247] + "..."

    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_candidates(path: str):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def rank_candidates(candidates_path: str, output_path: str):
    print(f"Loading candidates from {candidates_path}...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates):,} candidates")

    print("Scoring candidates...")
    results = []
    for i, c in enumerate(candidates):
        result = score_candidate(c)
        results.append(result)
        if (i + 1) % 10000 == 0:
            print(f"  Scored {i+1:,} / {len(candidates):,}")

    # Sort by composite score descending, then candidate_id ascending for ties
    results.sort(key=lambda r: (-r["composite"], r["candidate_id"]))

    # Take top 100
    top_100 = results[:100]

    print(f"\nTop 10 candidates:")
    for i, r in enumerate(top_100[:10]):
        p = r.get("profile", {})
        print(f"  {i+1:2d}. {r['candidate_id']}  {p.get('current_title','?'):35s}  "
              f"score={r['composite']:.4f}  "
              f"tc={r.get('tc_score',0):.2f} sk={r.get('sk_score',0):.2f} "
              f"bh={r['behavioral']:.2f}")

    # Honeypot check
    honeypots_in_top100 = sum(1 for r in top_100 if r.get("is_honeypot"))
    print(f"\nHoneypots in top 100: {honeypots_in_top100}")
    total_honeypots = sum(1 for r in results if r.get("is_honeypot"))
    print(f"Total honeypots detected: {total_honeypots:,}")

    # Build rows with proper non-increasing scores
    print(f"\nWriting submission to {output_path}...")
    rows = []
    for rank, result in enumerate(top_100, start=1):
        raw = result["composite"]
        max_score = top_100[0]["composite"]
        min_score = top_100[-1]["composite"]
        score_range = max(max_score - min_score, 1e-6)
        # Map to [0.10, 0.999]
        normalized = 0.10 + 0.899 * (raw - min_score) / score_range
        normalized = round(normalized, 4)
        rows.append([result["candidate_id"], rank, f"{normalized:.4f}",
                     generate_reasoning(result, rank)])

    # Ensure non-increasing
    for i in range(1, len(rows)):
        if float(rows[i][2]) > float(rows[i-1][2]):
            rows[i][2] = rows[i-1][2]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for row in rows:
            writer.writerow(row)

    print(f"Submission written: {output_path}")
    print(f"Score range: {rows[0][2]} → {rows[-1][2]}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Redrob Candidate Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl",
                        help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    import time
    t0 = time.time()
    rank_candidates(args.candidates, args.out)
    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()