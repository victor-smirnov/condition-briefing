"""The research plan: a fixed sequence of steps over a run record.

Each step reads earlier step outputs, writes one JSON record, and can be re-run.
A re-run never edits a finished run: `fork()` creates a child run that copies the
steps before the fork point and re-executes the rest with changed parameters.
"""

import copy
import difflib
import re
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Literal

from pydantic import BaseModel

from . import sources
from .config import settings
from .llm import Usage, structured
from .store import Store, make_store

STEPS = ["scope", "search_plan", "retrieve", "screen_extract", "aggregate", "synthesize", "verify"]

STEP_DOCS = {
    "scope": "LLM normalizes the condition: name, synonyms, MeSH terms, subtopics.",
    "search_plan": "LLM writes PubMed and ClinicalTrials.gov queries; user extras are appended.",
    "retrieve": "Code runs the queries and stores every source with the URL and time it was fetched.",
    "screen_extract": "LLM screens each source (include/exclude with a reason) and extracts evidence with verbatim quotes.",
    "aggregate": "Code counts included trials by phase, status, intervention class and sponsor.",
    "synthesize": "LLM writes the briefing; every statement cites evidence IDs.",
    "verify": "Code checks quotes against source text and citations against the evidence list.",
}


def default_params() -> dict:
    year = datetime.now(timezone.utc).year
    return {
        "scope": {"condition": "", "focus": "US health system strategy team", "instructions": ""},
        "search_plan": {
            "year_from": year - 5,
            "year_to": year,
            "pubmed_per_query": 8,
            "trials_per_query": 30,
            "instructions": "",
            "extra_pubmed_queries": [],  # PubMed query strings added verbatim
            "extra_trial_queries": [],  # {"condition": ..., "intervention": ...}
        },
        "retrieve": {},
        "screen_extract": {
            "instructions": "",
            "max_evidence_per_source": 3,
            "exclude_ids": [],  # user overrides of the LLM decision
            "include_ids": [],
        },
        "aggregate": {"top_n": 12},
        "synthesize": {"instructions": "", "audience": "health system strategy team"},
        "verify": {"near_match_threshold": 0.9, "support_check": True},
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def step_key(run_id: str, step: str) -> str:
    return f"{run_id}/steps/{STEPS.index(step):02d}_{step}.json"


def store() -> Store:
    return make_store(settings().store_url)


Progress = Callable[[str, str], None]


class Ctx:
    def __init__(self, st: Store, run: dict, on_progress: Progress) -> None:
        self.store = st
        self.run = run
        self.usage = Usage()
        self.progress = on_progress

    def params(self, step: str) -> dict:
        return self.run["params"][step]

    def out(self, step: str) -> dict:
        rec = self.store.read(step_key(self.run["id"], step))
        if rec is None or "outputs" not in rec:
            raise RuntimeError(f"step {step} has no output in run {self.run['id']}")
        return rec["outputs"]


# ---------------------------------------------------------------- step 1: scope

class Scope(BaseModel):
    condition: str
    synonyms: list[str]
    mesh_terms: list[str]
    subtopics: list[str]
    notes: str


SYS_SCOPE = """You scope a literature research task for a health system strategy team.
The team needs a briefing on a medical condition: current standard of care, emerging
treatments in development, and companies and institutions involved.
Return the canonical condition name, synonyms used in the literature, MeSH headings you
are confident exist, and 4 to 7 subtopics the briefing must cover. If the input is
ambiguous, pick the most common clinical meaning and say so in notes."""


def step_scope(ctx: Ctx) -> tuple[dict, str]:
    p = ctx.params("scope")
    user = f"Condition: {p['condition']}\nFocus: {p['focus']}\nExtra instructions: {p['instructions'] or 'none'}"
    s = structured(model=settings().model_main, system=SYS_SCOPE, user=user, schema=Scope,
                   usage=ctx.usage, prompt_id="scope.v1", effort="low")
    return s.model_dump(), f"{s.condition}; {len(s.synonyms)} synonyms, {len(s.subtopics)} subtopics"


# ---------------------------------------------------------- step 2: search plan

class PubmedQuery(BaseModel):
    purpose: str
    query: str


class TrialQuery(BaseModel):
    purpose: str
    condition: str
    intervention: str | None


class SearchPlan(BaseModel):
    pubmed: list[PubmedQuery]
    trials: list[TrialQuery]
    rationale: str


SYS_PLAN = """You write literature searches for a briefing on a medical condition for a
health system strategy team (standard of care, emerging treatments, companies and institutions).

PubMed: 3 to 5 queries in PubMed syntax using MeSH ([mh]) and title/abstract ([tiab]) terms.
- At least one query for current guidelines and consensus on management, filtered with
  (guideline[pt] OR practice guideline[pt] OR consensus development conference[pt] OR systematic review[pt]).
- At least one query for phase 3 results of recent therapies (clinical trial, phase iii[pt] or randomized controlled trial[pt]).
- At least one query for emerging or investigational therapies.
Do not add date limits; they are applied separately. Keep queries broad enough to return results.

ClinicalTrials.gov: 1 to 3 queries. "condition" is plain words. "intervention" is optional plain
words, null for a broad search. Phase 2/3, interventional and start-date filters are applied separately.

The user's extra instructions override these defaults."""


def step_search_plan(ctx: Ctx) -> tuple[dict, str]:
    p = ctx.params("search_plan")
    scope = ctx.out("scope")
    user = (f"Scope: {scope}\nDate range applied separately: {p['year_from']}-{p['year_to']}\n"
            f"Extra instructions: {p['instructions'] or 'none'}")
    plan = structured(model=settings().model_main, system=SYS_PLAN, user=user, schema=SearchPlan,
                      usage=ctx.usage, prompt_id="search_plan.v1", effort="low")
    pubmed = [{"purpose": q.purpose, "query": q.query, "origin": "llm"} for q in plan.pubmed]
    pubmed += [{"purpose": "user-added", "query": q, "origin": "user"} for q in p["extra_pubmed_queries"]]
    trials = [{"purpose": q.purpose, "condition": q.condition, "intervention": q.intervention, "origin": "llm"}
              for q in plan.trials]
    trials += [{"purpose": "user-added", "condition": q.get("condition") or scope["condition"],
                "intervention": q.get("intervention"), "origin": "user"} for q in p["extra_trial_queries"]]
    for i, q in enumerate(pubmed, 1):
        q["id"] = f"Q{i}"
    for i, q in enumerate(trials, 1):
        q["id"] = f"T{i}"
    out = {"pubmed": pubmed, "trials": trials, "rationale": plan.rationale,
           "year_from": p["year_from"], "year_to": p["year_to"]}
    return out, f"{len(pubmed)} PubMed queries, {len(trials)} trial queries"


# ------------------------------------------------------------- step 3: retrieve

def step_retrieve(ctx: Ctx) -> tuple[dict, str]:
    plan = ctx.out("search_plan")
    sp = ctx.params("search_plan")
    queries, found_by = [], defaultdict(list)
    srcs: dict[str, dict] = {}
    for q in plan["pubmed"]:
        rec = {"id": q["id"], "kind": "pubmed", "purpose": q["purpose"], "query": q["query"]}
        try:
            res = sources.pubmed_search(q["query"], plan["year_from"], plan["year_to"], sp["pubmed_per_query"])
            rec.update(url=res["url"], retrieved_at=res["retrieved_at"], total_hits=res["total_hits"],
                       returned=[f"PMID:{i}" for i in res["ids"]], translation=res["translation"])
            for i in res["ids"]:
                found_by[f"PMID:{i}"].append(q["id"])
        except Exception as e:
            rec.update(error=str(e), returned=[])
        queries.append(rec)
        ctx.progress("retrieve", f"{q['id']}: {rec.get('total_hits', 'error')} hits")
    pmids = [k.split(":")[1] for k in found_by]
    for i in range(0, len(pmids), 200):
        srcs.update(sources.pubmed_fetch(pmids[i:i + 200]))
    for q in plan["trials"]:
        rec = {"id": q["id"], "kind": "ctgov", "purpose": q["purpose"],
               "query": f"condition={q['condition']!r} intervention={q['intervention']!r}"}
        try:
            res = sources.ctgov_search(q["condition"], q["intervention"], plan["year_from"], sp["trials_per_query"])
            rec.update(url=res["url"], retrieved_at=res["retrieved_at"], total_hits=res["total_hits"],
                       returned=res["ids"])
            for nct, t in res["records"].items():
                found_by[nct].append(q["id"])
                srcs.setdefault(nct, t)
        except Exception as e:
            rec.update(error=str(e), returned=[])
        queries.append(rec)
        ctx.progress("retrieve", f"{q['id']}: {rec.get('total_hits', 'error')} hits")
    for sid, s in srcs.items():
        s["found_by"] = found_by[sid]
    n_art = sum(s["type"] == "article" for s in srcs.values())
    n_tr = len(srcs) - n_art
    errors = sum("error" in q for q in queries)
    summary = f"{len(queries)} queries ({errors} failed), {n_art} articles, {n_tr} trials"
    return {"queries": queries, "sources": srcs}, summary


# ------------------------------------------------------- step 4: screen/extract

Topic = Literal["standard_of_care", "emerging_treatment", "players", "epidemiology", "other"]


class EvidenceItem(BaseModel):
    claim: str
    quote: str
    topic: Topic


class Assessment(BaseModel):
    relevant: bool
    reason: str
    category: Literal["guideline", "review", "trial_result", "ongoing_trial", "other"]
    intervention_class: str | None
    evidence: list[EvidenceItem]


SYS_SCREEN = """You screen one source for a briefing for a health system strategy team. The
briefing covers current standard of care, emerging treatments in development, and companies
and institutions involved, for the condition named by the user.
Decide whether the source is relevant to that briefing and give a one-sentence reason.
If relevant, extract evidence items (at most the number the user gives). Rules:
- quote: one or two consecutive sentences copied verbatim from the source text. No ellipses,
  no paraphrase, no added or changed words.
- claim: one plain sentence for a strategy analyst stating what the quote supports.
- topic: the briefing section the evidence feeds.
- intervention_class: for trials and therapy papers, the drug or device class (for example
  "GLP-1 receptor agonist", "CAR-T cell therapy"); null otherwise.
If not relevant, return an empty evidence list."""


def step_screen_extract(ctx: Ctx) -> tuple[dict, str]:
    p = ctx.params("screen_extract")
    scope = ctx.out("scope")
    srcs = ctx.out("retrieve")["sources"]

    def assess(s: dict) -> tuple[str, Assessment]:
        user = (f"Condition: {scope['condition']} (synonyms: {', '.join(scope['synonyms'])})\n"
                f"Max evidence items: {p['max_evidence_per_source']}\n"
                f"Extra instructions: {p['instructions'] or 'none'}\n\n"
                f"Source {s['id']} ({s['type']}, {s['year']}):\n{s['text']}")
        return s["id"], structured(model=settings().model_fast, system=SYS_SCREEN, user=user, schema=Assessment,
                                   usage=ctx.usage, prompt_id="screen_extract.v1", effort="low")

    results: dict[str, Assessment] = {}
    failed: dict[str, str] = {}
    with ThreadPoolExecutor(settings().max_workers) as pool:
        futs = {pool.submit(assess, s): s["id"] for s in srcs.values()}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                sid, a = f.result()
                results[sid] = a
            except Exception as e:
                failed[futs[f]] = str(e)
            if n % 5 == 0 or n == len(futs):
                ctx.progress("screen_extract", f"screened {n}/{len(futs)}")

    assessments, evidence = {}, {}
    n = 0
    for sid in sorted(results):
        a = results[sid]
        decision, by = ("include" if a.relevant else "exclude"), "llm"
        if sid in p["exclude_ids"]:
            decision, by = "exclude", "user"
        elif sid in p["include_ids"]:
            decision, by = "include", "user"
        ev_ids = []
        if decision == "include":
            for e in a.evidence[: p["max_evidence_per_source"]]:
                n += 1
                eid = f"E{n:03d}"
                evidence[eid] = {"id": eid, "source_id": sid, **e.model_dump()}
                ev_ids.append(eid)
        assessments[sid] = {**a.model_dump(exclude={"evidence"}), "decision": decision, "decided_by": by,
                            "evidence_ids": ev_ids}
    for sid, err in failed.items():
        assessments[sid] = {"relevant": False, "reason": f"screening failed: {err}", "category": "other",
                            "intervention_class": None, "decision": "exclude", "decided_by": "error",
                            "evidence_ids": []}
    included = [s for s, a in assessments.items() if a["decision"] == "include"]
    excluded = [s for s, a in assessments.items() if a["decision"] == "exclude"]
    out = {"assessments": assessments, "evidence": evidence, "included": included, "excluded": excluded}
    summary = f"{len(included)} included, {len(excluded)} excluded, {len(evidence)} evidence items"
    if failed:
        summary += f", {len(failed)} failed"
    return out, summary


# ------------------------------------------------------------ step 5: aggregate

def step_aggregate(ctx: Ctx) -> tuple[dict, str]:
    top_n = ctx.params("aggregate")["top_n"]
    srcs = ctx.out("retrieve")["sources"]
    se = ctx.out("screen_extract")
    inc = [srcs[s] for s in se["included"]]
    trials = [s for s in inc if s["type"] == "trial"]
    articles = [s for s in inc if s["type"] == "article"]

    def group(items, key) -> dict[str, list[str]]:
        g = defaultdict(list)
        for s in items:
            g[key(s)].append(s["id"])
        return dict(sorted(g.items(), key=lambda kv: -len(kv[1])))

    sponsors: dict[str, dict] = {}
    for t in trials:
        m = t["meta"]
        entries = [(m["lead_sponsor"], m["lead_sponsor_class"], "lead")]
        entries += [(c["name"], c["class"], "collaborator") for c in m["collaborators"]]
        for name, cls, role in entries:
            d = sponsors.setdefault(name, {"name": name, "class": cls, "lead": [], "collaborator": []})
            d[role].append(t["id"])
    ranked = sorted(sponsors.values(), key=lambda d: -(len(d["lead"]) + len(d["collaborator"])))
    out = {
        "trials_by_phase": group(trials, lambda s: "/".join(s["meta"]["phases"]) or "NA"),
        "trials_by_status": group(trials, lambda s: s["meta"]["status"]),
        "trials_by_class": group(trials, lambda s: se["assessments"][s["id"]]["intervention_class"] or "unclassified"),
        "top_sponsors": ranked[:top_n],
        "industry_vs_other": dict(Counter("INDUSTRY" if t["meta"]["lead_sponsor_class"] == "INDUSTRY" else "OTHER"
                                          for t in trials)),
        "articles_by_year": dict(sorted(Counter(a["year"] for a in articles).items())),
        "articles_by_category": dict(Counter(se["assessments"][a["id"]]["category"] for a in articles)),
    }
    lead = ranked[0]["name"] if ranked else "none"
    return out, f"{len(trials)} trials, {len(articles)} articles; top sponsor: {lead}"


# ----------------------------------------------------------- step 6: synthesize

class Statement(BaseModel):
    text: str
    evidence_ids: list[str]
    basis: Literal["evidence", "inference"]


class Section(BaseModel):
    heading: str
    statements: list[Statement]


class Player(BaseModel):
    name: str
    kind: Literal["industry", "academic", "government", "other"]
    role: str
    trial_ids: list[str]


class Report(BaseModel):
    title: str
    executive_summary: list[Statement]
    standard_of_care: list[Section]
    emerging_treatments: list[Section]
    key_players: list[Player]
    implications: list[Statement]
    limitations: list[str]


SYS_SYNTH = """You write a structured briefing on a medical condition for a health system
strategy team. Use only the evidence items and trial data in the user message.
Rules:
- A statement with basis "evidence" cites 1 to 4 evidence IDs (E###) that directly support it.
  Never cite an ID that is not in the list.
- A statement with basis "inference" is your analysis for the strategy team. It may cite
  supporting evidence IDs, but it must not introduce drug names, numbers or dates that are
  absent from the evidence.
- executive_summary: 3 to 5 statements.
- standard_of_care: 2 to 4 sections (for example first-line therapy, escalation, procedures).
- emerging_treatments: 2 to 5 sections grouped by mechanism or intervention class. Mention
  phase and sponsor when the evidence gives them.
- key_players: 5 to 12 organizations. Copy names exactly as written in the sponsor data.
  trial_ids lists NCT IDs where the organization is lead sponsor or collaborator.
- implications: 3 to 6 statements on what this means for a health system (service lines,
  trial participation, partnerships, capacity, cost).
- limitations: what this evidence base does not cover.
Write plain, short sentences. No marketing language."""


def step_synthesize(ctx: Ctx) -> tuple[dict, str]:
    p = ctx.params("synthesize")
    scope = ctx.out("scope")
    srcs = ctx.out("retrieve")["sources"]
    se = ctx.out("screen_extract")
    agg = ctx.out("aggregate")

    ev_lines = []
    for eid, e in se["evidence"].items():
        s = srcs[e["source_id"]]
        cat = se["assessments"][s["id"]]["category"]
        ev_lines.append(f"{eid} [{e['topic']}] ({s['id']}, {s['year']}, {cat}) {e['claim']} | quote: {e['quote']}")
    trial_lines = []
    for sid in se["included"]:
        s = srcs[sid]
        if s["type"] != "trial":
            continue
        m = s["meta"]
        collab = ", ".join(c["name"] for c in m["collaborators"]) or "-"
        trial_lines.append(f"{sid} | {'/'.join(m['phases'])} | {m['status']} | lead: {m['lead_sponsor']} "
                           f"({m['lead_sponsor_class']}) | collaborators: {collab} | "
                           f"class: {se['assessments'][sid]['intervention_class']} | {'; '.join(m['interventions'])}")
    agg_brief = {k: v for k, v in agg.items() if k != "top_sponsors"}
    agg_brief = {k: ({kk: len(vv) for kk, vv in v.items()} if k.startswith("trials_by") else v)
                 for k, v in agg_brief.items()}
    sponsors = [f"{d['name']} ({d['class']}): lead {len(d['lead'])}, collaborator {len(d['collaborator'])}"
                for d in agg["top_sponsors"]]
    user = "\n".join([
        f"Condition: {scope['condition']}", f"Audience: {p['audience']}",
        f"Focus: {ctx.params('scope')['focus']}", f"Subtopics: {', '.join(scope['subtopics'])}",
        f"Extra instructions: {p['instructions'] or 'none'}", "",
        f"Aggregates: {agg_brief}", "Top sponsors:", *sponsors, "",
        "Included trials:", *trial_lines, "", "Evidence items:", *ev_lines,
    ])
    r = structured(model=settings().model_main, system=SYS_SYNTH, user=user, schema=Report,
                   usage=ctx.usage, prompt_id="synthesize.v1", effort="medium")
    rep = r.model_dump()
    n = 0

    def tag(stmts: list[dict]) -> None:
        nonlocal n
        for s in stmts:
            n += 1
            s["id"] = f"S{n}"

    tag(rep["executive_summary"])
    for part in ("standard_of_care", "emerging_treatments"):
        for sec in rep[part]:
            tag(sec["statements"])
    tag(rep["implications"])
    return rep, f"{n} statements, {len(rep['key_players'])} key players"


# --------------------------------------------------------------- step 7: verify

def _norm(t: str) -> str:
    t = t.lower().translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                           "–": "-", "—": "-", " ": " "}))
    return re.sub(r"\s+", " ", t).strip().rstrip(".")


def quote_match(quote: str, text: str, threshold: float) -> tuple[str, float]:
    """exact substring, near (most characters match in order), or missing."""
    q, t = _norm(quote), _norm(text)
    if q and q in t:
        return "exact", 1.0
    sm = difflib.SequenceMatcher(None, t, q, autojunk=False)
    score = sum(b.size for b in sm.get_matching_blocks()) / max(len(q), 1)
    return ("near" if score >= threshold else "missing"), round(score, 3)


def all_statements(rep: dict) -> list[tuple[str, dict]]:
    """(section label, statement) pairs in report order."""
    out = [("Executive summary", s) for s in rep["executive_summary"]]
    for part, label in (("standard_of_care", "Standard of care"), ("emerging_treatments", "Emerging treatments")):
        for sec in rep[part]:
            out += [(f"{label} / {sec['heading']}", s) for s in sec["statements"]]
    out += [("Implications", s) for s in rep["implications"]]
    return out


class SupportVerdict(BaseModel):
    id: str
    verdict: Literal["supported", "partial", "unsupported"]
    problem: str


class SupportBatch(BaseModel):
    results: list[SupportVerdict]


SYS_SUPPORT = """You check statements in a medical briefing against their cited evidence.
For each statement you get the quotes it cites and, for cited trials, the registry record.
Verdict "supported" if the quotes and records support every factual part of the statement,
"partial" if some factual part goes beyond them, "unsupported" if they do not support it.
For statements marked inference, judge only concrete facts (drug, number, date, organization,
guideline); analysis and recommendations are allowed. Give the problem in one short sentence,
empty if supported."""


def support_check(ctx: Ctx, rep: dict, se: dict, srcs: dict) -> dict[str, dict]:
    """An LLM entailment check: does each statement follow from what it cites?"""
    ev = se["evidence"]

    def item(s: dict) -> str:
        lines = [f"{s['id']} [{s['basis']}] {s['text']}"]
        cited = []
        for e in s["evidence_ids"]:
            if e in ev:
                lines.append(f"  quote {e} ({ev[e]['source_id']}): {ev[e]['quote']}")
                cited.append(ev[e]["source_id"])
        for sid in dict.fromkeys(cited):
            if srcs[sid]["type"] == "trial":
                m = srcs[sid]["meta"]
                lines.append(f"  record {sid}: {'/'.join(m['phases'])}, {m['status']}, lead {m['lead_sponsor']}, "
                             f"collaborators {', '.join(c['name'] for c in m['collaborators']) or '-'}, "
                             f"interventions {'; '.join(m['interventions'])}, conditions {', '.join(m['conditions'])}")
        return "\n".join(lines) if len(lines) > 1 else lines[0] + "\n  (no citations)"

    stmts = [s for _, s in all_statements(rep)]
    batches = [stmts[i:i + 12] for i in range(0, len(stmts), 12)]

    def judge(batch: list[dict]) -> list[SupportVerdict]:
        r = structured(model=settings().model_main, system=SYS_SUPPORT, user="\n\n".join(map(item, batch)),
                       schema=SupportBatch, usage=ctx.usage, prompt_id="verify_support.v1", effort="low")
        return r.results

    with ThreadPoolExecutor(4) as pool:
        verdicts = [v for b in pool.map(judge, batches) for v in b]
    return {v.id: {"verdict": v.verdict, "problem": v.problem} for v in verdicts}


def step_verify(ctx: Ctx) -> tuple[dict, str]:
    p = ctx.params("verify")
    thr = p["near_match_threshold"]
    srcs = ctx.out("retrieve")["sources"]
    se = ctx.out("screen_extract")
    rep = ctx.out("synthesize")
    ev_checks = {}
    for eid, e in se["evidence"].items():
        text = srcs[e["source_id"]]["text"]
        status, score = quote_match(e["quote"], text, thr)
        ev_checks[eid] = {"status": status, "score": score}
        if status == "near":  # keep the exact source passage next to the model's slightly edited quote
            q = e["quote"]
            m = difflib.SequenceMatcher(None, text, q, autojunk=False).find_longest_match(0, len(text), 0, len(q))
            start = max(0, m.a - m.b)  # align the quote's start with the longest shared run
            ev_checks[eid]["source_span"] = text[start: start + len(q) + 20]
    support = support_check(ctx, rep, se, srcs) if p.get("support_check", True) else {}
    st_checks = {}
    for _, s in all_statements(rep):
        issues = []
        sup = support.get(s["id"])
        if sup and sup["verdict"] == "unsupported":
            issues.append(f"not supported by its citations: {sup['problem']}")
        for eid in s["evidence_ids"]:
            if eid not in se["evidence"]:
                issues.append(f"cites unknown evidence {eid}")
            elif ev_checks[eid]["status"] == "missing":
                issues.append(f"cites {eid}, whose quote was not found in the source")
        if s["basis"] == "evidence" and not s["evidence_ids"]:
            issues.append("evidence statement without citations")
        st_checks[s["id"]] = {"ok": not issues, "issues": issues, "support": sup}
    pl_checks = {}
    included = set(se["included"])
    for pl in rep["key_players"]:
        issues = []
        for tid in pl["trial_ids"]:
            if tid not in included:
                issues.append(f"{tid} is not an included trial")
                continue
            m = srcs[tid]["meta"]
            names = [m["lead_sponsor"]] + [c["name"] for c in m["collaborators"]]
            if not any(_norm(pl["name"]) in _norm(n) or _norm(n) in _norm(pl["name"]) for n in names):
                issues.append(f"{pl['name']} is not a sponsor or collaborator of {tid}")
        if not pl["trial_ids"]:
            issues.append("no supporting trials")
        pl_checks[pl["name"]] = {"ok": not issues, "issues": issues}
    summary = {
        "evidence_exact": sum(c["status"] == "exact" for c in ev_checks.values()),
        "evidence_near": sum(c["status"] == "near" for c in ev_checks.values()),
        "evidence_missing": sum(c["status"] == "missing" for c in ev_checks.values()),
        "statements_ok": sum(c["ok"] for c in st_checks.values()),
        "statements_flagged": sum(not c["ok"] for c in st_checks.values()),
        "inference_statements": sum(s["basis"] == "inference" for _, s in all_statements(rep)),
        "players_ok": sum(c["ok"] for c in pl_checks.values()),
        "players_flagged": sum(not c["ok"] for c in pl_checks.values()),
        "support_supported": sum(v["verdict"] == "supported" for v in support.values()),
        "support_partial": sum(v["verdict"] == "partial" for v in support.values()),
        "support_unsupported": sum(v["verdict"] == "unsupported" for v in support.values()),
    }
    out = {"evidence_checks": ev_checks, "statement_checks": st_checks, "player_checks": pl_checks,
           "summary": summary}
    line = (f"quotes {summary['evidence_exact']} exact / {summary['evidence_near']} near / "
            f"{summary['evidence_missing']} missing; support {summary['support_supported']} full / "
            f"{summary['support_partial']} partial / {summary['support_unsupported']} none; "
            f"statements flagged {summary['statements_flagged']}; "
            f"players flagged {summary['players_flagged']}")
    return out, line


STEP_FNS = {
    "scope": step_scope, "search_plan": step_search_plan, "retrieve": step_retrieve,
    "screen_extract": step_screen_extract, "aggregate": step_aggregate,
    "synthesize": step_synthesize, "verify": step_verify,
}


# ------------------------------------------------------------- run management

def _noop(step: str, msg: str) -> None:
    pass


def load_run(run_id: str, st: Store | None = None) -> dict:
    run = (st or store()).read(f"{run_id}/run.json")
    if run is None:
        raise KeyError(f"no run {run_id}")
    return run


def load_step(run_id: str, step: str, st: Store | None = None) -> dict | None:
    return (st or store()).read(step_key(run_id, step))


def list_runs(st: Store | None = None) -> list[dict]:
    st = st or store()
    runs = [st.read(f"{r}/run.json") for r in st.list_runs()]
    return sorted((r for r in runs if r), key=lambda r: r["created_at"], reverse=True)


def _new_id(condition: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", condition.lower()).strip("-")[:24]
    return f"{slug}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def create_run(condition: str, focus: str | None = None, param_changes: dict | None = None) -> str:
    st = store()
    params = default_params()
    params["scope"]["condition"] = condition
    if focus:
        params["scope"]["focus"] = focus
    for step, ch in (param_changes or {}).items():
        params[step].update(ch)
    run = {
        "id": _new_id(condition), "created_at": now_iso(), "title": condition, "parent": None,
        "change_note": None, "params": params, "status": "created", "error": None,
        "steps": {s: {"status": "pending"} for s in STEPS},
    }
    st.write(f"{run['id']}/run.json", run)
    return run["id"]


def fork(run_id: str, from_step: str, param_changes: dict, note: str = "") -> tuple[str, str]:
    """Create a child run. Returns (new_run_id, actual start step).

    The start step is the earlier of `from_step` and the first step whose params changed,
    because a parameter change invalidates that step and everything after it.
    """
    st = store()
    parent = load_run(run_id, st)
    # Parameters added after the parent was created take their defaults.
    params = {s: {**d, **parent["params"].get(s, {})} for s, d in default_params().items()}
    start = STEPS.index(from_step)
    for step, ch in param_changes.items():
        if step not in params:
            raise ValueError(f"unknown step {step}; steps are {STEPS}")
        unknown = set(ch) - set(params[step])
        if unknown:
            raise ValueError(f"unknown params for {step}: {sorted(unknown)}; known: {sorted(params[step])}")
        params[step].update(ch)
        start = min(start, STEPS.index(step))
    child = {
        "id": _new_id(parent["title"]), "created_at": now_iso(), "title": parent["title"],
        "parent": {"run_id": run_id, "fork_step": STEPS[start]}, "change_note": note or None,
        "params": params, "status": "created", "error": None,
        "steps": {s: {"status": "pending"} for s in STEPS},
    }
    for s in STEPS[:start]:
        rec = load_step(run_id, s, st)
        if rec is None:
            raise RuntimeError(f"parent run {run_id} has no output for step {s}")
        rec = {**rec, "run_id": child["id"], "inherited_from": rec.get("inherited_from") or run_id}
        st.write(step_key(child["id"], s), rec)
        child["steps"][s] = {**parent["steps"][s], "status": "inherited",
                             "inherited_from": rec["inherited_from"]}
    st.write(f"{child['id']}/run.json", child)
    return child["id"], STEPS[start]


def execute(run_id: str, start_step: str | None = None, on_progress: Progress = _noop) -> dict:
    st = store()
    run = load_run(run_id, st)
    start = STEPS.index(start_step) if start_step else next(
        (i for i, s in enumerate(STEPS) if run["steps"][s]["status"] not in ("done", "inherited")), len(STEPS))
    run["status"], run["error"] = "running", None
    st.write(f"{run_id}/run.json", run)
    for step in STEPS[start:]:
        ctx = Ctx(st, run, on_progress)
        t0 = time.time()
        run["steps"][step] = {"status": "running", "started_at": now_iso()}
        st.write(f"{run_id}/run.json", run)
        on_progress(step, "started")
        try:
            outputs, summary = STEP_FNS[step](ctx)
        except Exception as e:
            run["steps"][step].update(status="failed", error=str(e), finished_at=now_iso())
            run["status"], run["error"] = "failed", f"{step}: {e}"
            st.write(f"{run_id}/run.json", run)
            raise
        meta = {"duration_s": round(time.time() - t0, 1), "llm": ctx.usage.as_dict()}
        st.write(step_key(run_id, step), {
            "run_id": run_id, "step": step, "index": STEPS.index(step), "params": run["params"][step],
            "summary": summary, "meta": meta, "inherited_from": None, "outputs": outputs,
        })
        run["steps"][step].update(status="done", finished_at=now_iso(), summary=summary, **meta)
        st.write(f"{run_id}/run.json", run)
        on_progress(step, f"done: {summary}")
    run["status"] = "done"
    st.write(f"{run_id}/run.json", run)
    return run


def diff_runs(a: str, b: str) -> dict:
    """What changed from run a to run b."""
    st = store()
    out: dict = {"from": a, "to": b}
    sa, sb = load_step(a, "screen_extract", st), load_step(b, "screen_extract", st)
    if sa and sb:
        ia, ib = set(sa["outputs"]["included"]), set(sb["outputs"]["included"])
        out["sources_added"] = sorted(ib - ia)
        out["sources_removed"] = sorted(ia - ib)
        out["evidence_count"] = [len(sa["outputs"]["evidence"]), len(sb["outputs"]["evidence"])]
    ra, rb = load_step(a, "synthesize", st), load_step(b, "synthesize", st)
    if ra and rb:
        def by_section(rep):
            g = defaultdict(list)
            for label, s in all_statements(rep["outputs"]):
                g[label.split(" / ")[0]].append(s["text"])
            return g
        ga, gb = by_section(ra), by_section(rb)
        out["sections"] = {
            k: ("unchanged" if ga.get(k) == gb.get(k) else
                f"changed ({len(ga.get(k, []))} -> {len(gb.get(k, []))} statements)")
            for k in sorted(set(ga) | set(gb))
        }
        pa = {p["name"] for p in ra["outputs"]["key_players"]}
        pb = {p["name"] for p in rb["outputs"]["key_players"]}
        out["players_added"], out["players_removed"] = sorted(pb - pa), sorted(pa - pb)
    va, vb = load_step(a, "verify", st), load_step(b, "verify", st)
    if va and vb:
        out["verify"] = {"from": va["outputs"]["summary"], "to": vb["outputs"]["summary"]}
    return out
