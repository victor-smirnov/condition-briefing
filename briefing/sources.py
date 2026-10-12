"""Clients for PubMed (NCBI E-utilities) and ClinicalTrials.gov API v2.

Every search returns the exact URL it called and the time of the call, so the
retrieve step can record where each source came from.
"""

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx

from .config import settings

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CTGOV = "https://clinicaltrials.gov/api/v2/studies"
TRIAL_STATUSES = "RECRUITING,ACTIVE_NOT_RECRUITING,NOT_YET_RECRUITING,COMPLETED"

_http = httpx.Client(timeout=30, headers={"User-Agent": "briefing-demo/0.1"})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ncbi_params(**kw) -> dict:
    s = settings()
    p = {"tool": "briefing-demo", "email": s.ncbi_email, **kw}
    if s.ncbi_api_key:
        p["api_key"] = s.ncbi_api_key
    return p


def _get(url: str, params: dict) -> httpx.Response:
    for attempt in range(3):
        r = _http.get(url, params=params)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(1 + attempt)
    r.raise_for_status()
    return r


def pubmed_search(query: str, year_from: int, year_to: int, retmax: int) -> dict:
    params = _ncbi_params(
        db="pubmed", term=query, retmode="json", retmax=retmax, sort="relevance",
        datetype="pdat", mindate=str(year_from), maxdate=str(year_to),
    )
    r = _get(f"{EUTILS}/esearch.fcgi", params)
    res = r.json()["esearchresult"]
    time.sleep(0.35)  # NCBI allows 3 requests/s without an API key
    return {
        "url": str(r.url),
        "retrieved_at": now_iso(),
        "total_hits": int(res.get("count", 0)),
        "ids": res.get("idlist", []),
        "translation": res.get("querytranslation", ""),
    }


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def pubmed_fetch(pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    r = _http.post(f"{EUTILS}/efetch.fcgi", data=_ncbi_params(db="pubmed", id=",".join(pmids), retmode="xml"))
    r.raise_for_status()
    fetched_at = now_iso()
    out = {}
    for art in ET.fromstring(r.content).iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        pmid = mc.findtext("PMID")
        a = mc.find("Article")
        parts = []
        for ab in a.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
        year = a.findtext("Journal/JournalIssue/PubDate/Year") or (
            a.findtext("Journal/JournalIssue/PubDate/MedlineDate") or ""
        )[:4]
        authors = a.findall("AuthorList/Author")
        first = authors[0] if authors else None
        doi = next(
            (i.text for i in art.findall("PubmedData/ArticleIdList/ArticleId") if i.get("IdType") == "doi"),
            None,
        )
        title = _text(a.find("ArticleTitle"))
        abstract = "\n".join(parts)
        out[f"PMID:{pmid}"] = {
            "id": f"PMID:{pmid}",
            "type": "article",
            "title": title,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "year": year,
            "text": f"{title}\n{abstract}",
            "retrieved_at": fetched_at,
            "meta": {
                "journal": a.findtext("Journal/Title") or "",
                "pub_types": [_text(p) for p in a.findall("PublicationTypeList/PublicationType")],
                "first_author": (
                    f"{first.findtext('LastName') or ''} {first.findtext('Initials') or ''}".strip()
                    if first is not None else ""
                ),
                "first_affiliation": _text(first.find("AffiliationInfo/Affiliation")) if first is not None else "",
                "doi": doi,
                "has_abstract": bool(parts),
            },
        }
    return out


def _trial_record(study: dict, fetched_at: str) -> dict:
    p = study["protocolSection"]
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    sponsors = p.get("sponsorCollaboratorsModule", {})
    design = p.get("designModule", {})
    interventions = p.get("armsInterventionsModule", {}).get("interventions", [])
    nct = ident["nctId"]
    lead = sponsors.get("leadSponsor", {})
    collaborators = sponsors.get("collaborators", [])
    phases = design.get("phases", [])
    conditions = p.get("conditionsModule", {}).get("conditions", [])
    start = status.get("startDateStruct", {}).get("date", "")
    intr = [f"{i.get('type', '')}: {i.get('name', '')}" for i in interventions]
    text = "\n".join([
        ident.get("briefTitle", ""),
        f"Official title: {ident.get('officialTitle', '')}",
        f"Conditions: {', '.join(conditions)}",
        f"Interventions: {'; '.join(intr)}",
        f"Phase: {', '.join(phases)}",
        f"Status: {status.get('overallStatus', '')}",
        f"Start date: {start}",
        f"Lead sponsor: {lead.get('name', '')}",
        f"Collaborators: {', '.join(c['name'] for c in collaborators)}",
        f"Summary: {' '.join(p.get('descriptionModule', {}).get('briefSummary', '').split())}",
    ])
    return {
        "id": nct,
        "type": "trial",
        "title": ident.get("briefTitle", ""),
        "url": f"https://clinicaltrials.gov/study/{nct}",
        "year": start[:4],
        "text": text,
        "retrieved_at": fetched_at,
        "meta": {
            "phases": phases,
            "status": status.get("overallStatus", ""),
            "start_date": start,
            "lead_sponsor": lead.get("name", ""),
            "lead_sponsor_class": lead.get("class", ""),
            "collaborators": [{"name": c["name"], "class": c.get("class", "")} for c in collaborators],
            "interventions": intr,
            "conditions": conditions,
            "enrollment": design.get("enrollmentInfo", {}).get("count"),
        },
    }


def ctgov_search(condition: str, intervention: str | None, year_from: int, limit: int) -> dict:
    params = {
        "query.cond": condition,
        "filter.overallStatus": TRIAL_STATUSES,
        "filter.advanced": (
            "AREA[StudyType]INTERVENTIONAL AND (AREA[Phase]PHASE2 OR AREA[Phase]PHASE3) "
            f"AND AREA[StartDate]RANGE[{year_from}-01-01,MAX]"
        ),
        "pageSize": limit,
        "countTotal": "true",
    }
    if intervention:
        params["query.intr"] = intervention
    r = _get(CTGOV, params)
    data = r.json()
    fetched_at = now_iso()
    trials = [_trial_record(s, fetched_at) for s in data.get("studies", [])]
    return {
        "url": str(r.url),
        "retrieved_at": fetched_at,
        "total_hits": data.get("totalCount", len(trials)),
        "ids": [t["id"] for t in trials],
        "records": {t["id"]: t for t in trials},
    }
