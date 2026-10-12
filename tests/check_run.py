"""Independent checks of a finished run. Does not reuse the app's parsers or verify step.

    uv run --env-file .env python tests/check_run.py <run_id> [--judge-model gpt-5.4] [--data data/runs]

1. Every cited source is fetched again live (PubMed efetch text, ClinicalTrials.gov JSON) and every
   cited quote is searched in the live text.
2. Every key player is checked against the live sponsor and collaborator fields of its trials.
3. Included sources fall inside the run's date range.
4. Trial counts by phase are recomputed from the stored source records.
5. A different model judges whether each statement is supported by the quotes it cites.
"""

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from openai import OpenAI

http = httpx.Client(timeout=30, headers={"User-Agent": "briefing-check/0.1"})


def norm(t: str) -> str:
    t = t.lower().translate(str.maketrans("‘’“”–—− ", "''\"\"--- "))
    return re.sub(r"[^a-z0-9%<>=.,;:()+/-]+", " ", t).strip(" .")


# Field labels the app adds when it builds source text ("Lead sponsor: ...", "CONCLUSIONS: ...").
# They are structure in the live XML/JSON, not text, so a quote is split on them.
LABEL = re.compile(r"(?:^|\s)(?:[A-Z][A-Z ]{2,30}|Official title|Conditions|Interventions|Phase|Status|"
                   r"Start date|Lead sponsor|Collaborators|Summary):\s")


def quote_in(quote: str, text: str) -> bool:
    t = norm(text)
    parts = [norm(p) for p in LABEL.split(quote) if len(norm(p)) > 3]
    return bool(parts) and all(p in t for p in parts)


def flatten(obj) -> str:
    if isinstance(obj, dict):
        return " ".join(flatten(v) for v in obj.values())
    if isinstance(obj, list):
        return " ".join(flatten(v) for v in obj)
    return str(obj)


def live_pubmed(pmids: list[str]) -> dict[str, tuple[str, None]]:
    """One batched efetch (NCBI allows 3 requests/s); all text of each article, no field parsing."""
    import xml.etree.ElementTree as ET

    out = {}
    for i in range(0, len(pmids), 150):
        r = http.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                      data={"db": "pubmed", "id": ",".join(pmids[i:i + 150]), "retmode": "xml"})
        r.raise_for_status()
        for art in ET.fromstring(r.content).iter("PubmedArticle"):
            out[f"PMID:{art.findtext('MedlineCitation/PMID')}"] = ("".join(art.itertext()), None)
        time.sleep(0.5)
    return out


def live_trial(sid: str) -> tuple[str, dict]:
    r = http.get(f"https://clinicaltrials.gov/api/v2/studies/{sid}")
    r.raise_for_status()
    study = r.json()
    return flatten(study), study


JUDGE = """You check a medical briefing. For each statement you get the quotes it cites.
Verdict "supported" if the quotes support every factual part of the statement,
"partial" if some part goes beyond the quotes, "unsupported" if the quotes do not support it.
For statements marked inference, judge only whether any concrete fact (drug, number, date,
organization) in it is absent from the quotes; analysis and recommendations are allowed.
Return JSON: {"results": [{"id": "S1", "verdict": "...", "problem": "short reason or empty"}]}"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--data", default="data/runs")
    ap.add_argument("--judge-model", default="gpt-5.4")
    ap.add_argument("--no-judge", action="store_true", help="skip check 5 (needs the OpenAI API)")
    a = ap.parse_args()
    root = Path(a.data) / a.run_id
    step = lambda n, s: json.loads((root / "steps" / f"{n:02d}_{s}.json").read_text())["outputs"]
    plan, ret, se, agg, rep = (step(1, "search_plan"), step(2, "retrieve"), step(3, "screen_extract"),
                               step(4, "aggregate"), step(5, "synthesize"))
    srcs, ev = ret["sources"], se["evidence"]
    stmts = rep["executive_summary"] + [s for p in ("standard_of_care", "emerging_treatments")
                                        for sec in rep[p] for s in sec["statements"]] + rep["implications"]
    problems: list[str] = []

    # 1. live re-fetch and quote search
    cited_ev = sorted({e for s in stmts for e in s["evidence_ids"] if e in ev})
    cited_src = sorted({ev[e]["source_id"] for e in cited_ev} | {t for p in rep["key_players"] for t in p["trial_ids"]})
    trials = [s for s in cited_src if not s.startswith("PMID:")]
    live = live_pubmed([s[5:] for s in cited_src if s.startswith("PMID:")])
    with ThreadPoolExecutor(4) as pool:
        live |= dict(zip(trials, pool.map(live_trial, trials)))
    found = 0
    for e in cited_ev:
        q, sid = ev[e]["quote"], ev[e]["source_id"]
        if sid not in live:
            problems.append(f"{sid} could not be fetched live")
        elif quote_in(q, live[sid][0]):
            found += 1
        else:
            problems.append(f"quote {e} not found in live {sid}: {q[:90]}")
    print(f"1. live quotes: {found}/{len(cited_ev)} cited quotes found in {len(cited_src)} re-fetched sources")

    # 2. key players against live sponsor data
    ok_players = 0
    for p in rep["key_players"]:
        bad = []
        for t in p["trial_ids"]:
            study = live.get(t, (None, None))[1]
            if not study:
                bad.append(f"{t} not fetched")
                continue
            sc = study["protocolSection"].get("sponsorCollaboratorsModule", {})
            names = [sc.get("leadSponsor", {}).get("name", "")] + [c["name"] for c in sc.get("collaborators", [])]
            if not any(norm(p["name"]) in norm(n) or norm(n) in norm(p["name"]) for n in names):
                bad.append(f"{t} sponsors are {names}")
        if bad or not p["trial_ids"]:
            problems.append(f"player {p['name']}: {bad or 'no trials'}")
        else:
            ok_players += 1
    print(f"2. key players: {ok_players}/{len(rep['key_players'])} match live sponsor data")

    # 3. date range
    lo, hi = plan["year_from"], plan["year_to"]
    out_of_range = [s for s in se["included"] if not (srcs[s]["year"].isdigit() and lo <= int(srcs[s]["year"]) <= hi)]
    for s in out_of_range:
        problems.append(f"{s} year {srcs[s]['year']!r} outside {lo}-{hi}")
    print(f"3. date range {lo}-{hi}: {len(se['included']) - len(out_of_range)}/{len(se['included'])} included sources inside")

    # 4. recount phases
    recount = Counter("/".join(srcs[s]["meta"]["phases"]) or "NA" for s in se["included"] if srcs[s]["type"] == "trial")
    stored = {k: len(v) for k, v in agg["trials_by_phase"].items()}
    same = dict(recount) == stored
    if not same:
        problems.append(f"phase counts differ: recount {dict(recount)} vs stored {stored}")
    print(f"4. trial phase counts: {'match' if same else 'differ'} {dict(recount)}")

    # 5. independent judge
    def judge(batch: list[dict]) -> list[dict]:
        items = []
        for s in batch:
            quotes = [f"{e} ({ev[e]['source_id']}): {ev[e]['quote']}" for e in s["evidence_ids"] if e in ev]
            items.append(f"{s['id']} [{s['basis']}] {s['text']}\n  " + ("\n  ".join(quotes) or "(no quotes)"))
        r = OpenAI().chat.completions.create(
            model=a.judge_model, response_format={"type": "json_object"}, reasoning_effort="low",
            messages=[{"role": "system", "content": JUDGE}, {"role": "user", "content": "\n\n".join(items)}])
        return json.loads(r.choices[0].message.content)["results"]

    batches = [] if a.no_judge else [stmts[i:i + 12] for i in range(0, len(stmts), 12)]
    with ThreadPoolExecutor(4) as pool:
        verdicts = [v for b in pool.map(judge, batches) for v in b]
    c = Counter(v["verdict"] for v in verdicts)
    by_basis = {s["id"]: s["basis"] for s in stmts}
    for v in verdicts:
        if v["verdict"] != "supported":
            problems.append(f"judge {v['id']} ({by_basis.get(v['id'])}) {v['verdict']}: {v['problem']}")
    print(f"5. judge ({a.judge_model}): {dict(c)} over {len(stmts)} statements")

    print(f"\n{len(problems)} problems")
    for p in problems:
        print(" -", p)


if __name__ == "__main__":
    main()
