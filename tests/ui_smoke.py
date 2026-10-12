"""Browser test of the main user scenarios against a running app, in the order a user meets them.

    uv run python tests/ui_smoke.py [--url URL] [--shots DIR] [--condition "psoriasis"] [--skip-llm]

1. Open the app: a report is visible.            5. Ask the assistant to re-run a step.
2. Check a citation: link and quote tooltip.      6. Switch runs in the sidebar.
3. Start research on a new condition.             7. Review plan, evidence, downloads.
4. Ask the assistant a provenance question.

Scenarios 3 to 5 call the LLM and take about 4 minutes; --skip-llm leaves them out.
Prints one line per check and exits non-zero if any check fails.
"""

import argparse
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> bool:
    results.append((ok, what))
    print(("PASS " if ok else "FAIL ") + what, flush=True)
    return ok


def settle(page: Page, timeout: int = 60_000) -> None:
    """Wait until Streamlit finished the current script run."""
    page.wait_for_timeout(400)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=timeout)
    page.wait_for_timeout(300)


def no_exceptions(page: Page, where: str) -> None:
    n = page.locator('[data-testid="stException"]').count()
    check(n == 0, f"{where}: no Streamlit exceptions ({n})")


def run_label(page: Page) -> str:
    return page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"] input').input_value()


def wait_answer(page: Page, before: int, timeout: int = 300_000) -> str:
    """Wait until the assistant's final message replaced the 'Working' placeholder."""
    page.wait_for_function(
        "n => { const m = document.querySelectorAll('[data-testid=\"stChatMessage\"]');"
        " return m.length >= n && !m[m.length - 1].innerText.includes('Working'); }", arg=before + 2, timeout=timeout)
    settle(page, timeout)
    return page.locator('[data-testid="stChatMessage"]').last.inner_text()


def ask(page: Page, text: str, timeout: int = 300_000) -> str:
    before = page.locator('[data-testid="stChatMessage"]').count()
    box = page.locator('[data-testid="stChatInput"] textarea')
    box.fill(text)
    box.press("Enter")
    return wait_answer(page, before, timeout)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--shots", default="ui-shots")
    ap.add_argument("--condition", default="psoriasis")
    ap.add_argument("--skip-llm", action="store_true")
    a = ap.parse_args()
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000}, accept_downloads=True)
        main_area = page.locator('[data-testid="stMain"]')
        side = page.locator('[data-testid="stSidebar"]')

        # 1. Open the app
        t0 = time.time()
        page.goto(a.url)
        main_area.get_by_role("heading", name="Executive summary").wait_for(timeout=60_000)
        check(True, f"1. report visible {time.time() - t0:.1f}s after opening")
        no_exceptions(page, "1. load")
        check(main_area.get_by_text(re.compile(r"Verification: \d+ statements verified")).count() > 0,
              "1. verification banner at top of report")
        page.screenshot(path=shots / "01_open.png")

        # 2. Citation
        cite = main_area.get_by_role("link", name=re.compile(r"^\[\d+\]$")).first
        href, title = cite.get_attribute("href") or "", cite.get_attribute("title") or ""
        check(bool(re.match(r"https://(pubmed\.ncbi\.nlm\.nih\.gov|clinicaltrials\.gov)/", href)),
              f"2. citation links to the source ({href})")
        check("Quote:" in title, f"2. citation tooltip shows the quote ({title[:70]}...)")

        if not a.skip_llm:
            # 3. New research
            side.get_by_label("Condition").fill(a.condition)
            side.get_by_role("button", name="Start research").click()
            t0 = time.time()
            status = side.locator('[data-testid="stExpander"]').first
            page.wait_for_function(
                "() => /Step \\d\\/7/.test(document.querySelector('[data-testid=\"stSidebar\"]').innerText)",
                timeout=30_000)
            check(True, "3. progress shows 'Step n/7' while running")
            page.screenshot(path=shots / "03a_running.png")
            page.wait_for_function(
                "c => document.querySelector('[data-testid=\"stSidebar\"] [data-testid=\"stSelectbox\"] input')"
                "?.value.toLowerCase().includes(c)", arg=a.condition.lower(), timeout=400_000)
            settle(page)
            check(True, f"3. new run finished and selected in {time.time() - t0:.0f}s")
            check(main_area.get_by_role("heading", name="Executive summary").count() > 0, "3. new report shown")
            no_exceptions(page, "3. after new run")
            page.screenshot(path=shots / "03b_new_report.png", full_page=True)

            # 4. Provenance question via a suggestion button
            t0 = time.time()
            main_area.get_by_role("button", name=re.compile("Why is S2 in the report")).click()
            ans = wait_answer(page, 0)
            check("Error:" not in ans and len(ans) > 100, f"4. suggested question answered in {time.time() - t0:.0f}s")
            check(bool(re.search(r"E\d{3}", ans)) and bool(re.search(r"PMID:\d+|NCT\d{8}", ans)),
                  "4. answer cites evidence and source IDs")
            check(page.locator('[data-testid="stChatMessage"]').last.get_by_text(re.compile(r"\d+ tool calls")).count() > 0,
                  "4. answer lists its tool calls")
            page.screenshot(path=shots / "04_chat.png")

            # 5. Re-run through the assistant
            before = run_label(page)
            t0 = time.time()
            ans = ask(page, "Re-run only the synthesis step so the implications focus on rural hospitals.")
            after = run_label(page)
            check(after != before and "re-run from synthesize" in after,
                  f"5. re-run created and selected a child run in {time.time() - t0:.0f}s ({after})")
            report_col = main_area.locator('[data-testid="stColumn"]').first
            check("rural" in report_col.inner_text().lower(), "5. new report mentions rural hospitals")
            check(side.get_by_text("re-run from synthesize").count() > 0, "5. lineage shows the re-run")
            no_exceptions(page, "5. after re-run")
            page.screenshot(path=shots / "05_rerun.png")

        # 6. Switch runs
        before = run_label(page)
        side.locator('[data-testid="stSelectbox"]').click()
        opts = page.get_by_role("option")
        texts = [opts.nth(i).inner_text() for i in range(opts.count())]
        target = next(i for i, t in enumerate(texts) if t.strip() != before.strip() and "[" not in t)
        opts.nth(target).click()
        settle(page)
        check(run_label(page).strip() == texts[target].strip(), f"6. switched run to '{texts[target][:60]}'")
        check(main_area.get_by_role("heading", name="Executive summary").count() > 0, "6. report of the other run shown")
        no_exceptions(page, "6. after switch")

        # 7. Plan, evidence, downloads
        page.get_by_role("tab", name="Plan and steps").click()
        exps = main_area.locator('[data-testid="stColumn"]').first.locator('[data-testid="stExpander"]')
        check(exps.count() == 7, f"7. plan shows 7 steps ({exps.count()})")
        for i in range(exps.count()):
            exps.nth(i).locator("summary").click()
        settle(page)
        no_exceptions(page, "7. plan expanded")
        page.screenshot(path=shots / "07_plan.png", full_page=True)
        page.get_by_role("tab", name="Sources and evidence").click()
        settle(page)
        check(main_area.locator('[data-testid="stDataFrame"]').count() > 0, "7. evidence table rendered")
        no_exceptions(page, "7. evidence tab")
        page.get_by_role("tab", name="Report").click()
        for label in ("Download report (Markdown)", "Download provenance bundle (zip)"):
            with page.expect_download(timeout=15_000) as d:
                page.get_by_role("button", name=label).click()
            check(Path(d.value.path()).stat().st_size > 1000, f"7. download '{d.value.suggested_filename}'")

        browser.close()
    failed = [w for ok, w in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
