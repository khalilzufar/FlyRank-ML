"""A small, safe, tool-calling personal portfolio agent.

It intentionally uses only the Python standard library so a reviewer can run it
from a fresh clone. The optional HTTP tool reads a public page and never writes
to the network.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Case:
    case_id: str
    title: str
    problem: str
    result: str
    evidence: str
    next_step: str


CASES = (
    Case(
        case_id="flyrank-refresh",
        title="Content refresh priority",
        problem="Editorial teams need a ranked review queue for pages with observed traffic decline.",
        result="The grouped holdout model reached Precision@50 0.580, matching the transparent baseline.",
        evidence="111,133 rows across 49 pseudonymous client groups; grouped holdout is primary.",
        next_step="Add a time-aware future-window check before treating the ranking as a durable signal.",
    ),
)


def search_cases(query: str) -> list[dict[str, Any]]:
    """Search only the public-safe case registry."""
    needle = query.strip().lower()
    matches = [c for c in CASES if not needle or needle in json.dumps(asdict(c)).lower()]
    return [asdict(c) for c in matches]


def recommend_next_step(case_id: str) -> dict[str, Any]:
    for case in CASES:
        if case.case_id == case_id:
            return {"case_id": case.case_id, "recommendation": case.next_step, "basis": case.evidence}
    return {"case_id": case_id, "recommendation": "Not verified: select a known public case first.", "basis": "none"}


class TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.in_title = tag.lower() == "title"

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.parts.append(data.strip())


def fetch_public_page(url: str) -> dict[str, Any]:
    """Read one public page; this tool has no write capability."""
    if not re.fullmatch(r"https?://[^\s]+", url):
        return {"url": url, "status": "blocked", "reason": "Only public HTTP(S) URLs are allowed."}
    request = Request(url, headers={"User-Agent": "FlyRank-portfolio-agent/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            body = response.read(200_000)
            parser = TitleParser()
            parser.feed(body.decode("utf-8", errors="replace"))
            return {"url": url, "status": response.status, "title": "".join(parser.parts), "bytes_read": len(body)}
    except HTTPError as exc:
        return {"url": url, "status": exc.code, "title": "", "bytes_read": 0}
    except (URLError, TimeoutError, ValueError) as exc:
        return {"url": url, "status": "unavailable", "reason": type(exc).__name__}


Tool = Callable[..., Any]
TOOLS: dict[str, Tool] = {
    "search_cases": search_cases,
    "recommend_next_step": recommend_next_step,
    "fetch_public_page": fetch_public_page,
}


def run_agent(query: str, site_url: str | None = None) -> dict[str, Any]:
    """Classify a question and call the minimum tools needed to answer it."""
    q = query.lower()
    calls: list[dict[str, Any]] = []
    matches = TOOLS["search_cases"](query)
    calls.append({"tool": "search_cases", "input": query, "output": matches})
    if not matches:
        # A planning question such as “what next?” may not contain a case
        # keyword. Fall back to the small public registry, never to private
        # context or an invented record.
        matches = TOOLS["search_cases"]("")
        calls.append({"tool": "search_cases", "input": "public registry fallback", "output": matches})
    if matches:
        recommendation = TOOLS["recommend_next_step"](matches[0]["case_id"])
        calls.append({"tool": "recommend_next_step", "input": matches[0]["case_id"], "output": recommendation})
    else:
        recommendation = {"recommendation": "Not verified: no matching public case."}
    if site_url or any(word in q for word in ("check", "live", "url", "paper", "site")):
        target = site_url or "https://khalilzufar.github.io/FlyRank-ML/"
        page = TOOLS["fetch_public_page"](target)
        calls.append({"tool": "fetch_public_page", "input": target, "output": page})
    answer = {
        "answer": recommendation.get("recommendation", "No recommendation available."),
        "evidence": matches[0]["evidence"] if matches else "No matching evidence.",
        "calls": calls,
        "boundary": "Decision support only; a human still decides what to change.",
    }
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public-safe FlyRank portfolio agent")
    parser.add_argument("--query", required=True)
    parser.add_argument("--site-url")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = run_agent(args.query, args.site_url)
    if args.as_json:
        print(json.dumps(result, indent=2))
        return
    print("Answer:", result["answer"])
    print("Evidence:", result["evidence"])
    for call in result["calls"]:
        print(f"Tool {call['tool']}: {json.dumps(call['output'], ensure_ascii=False)}")
    print("Boundary:", result["boundary"])


if __name__ == "__main__":
    main()
