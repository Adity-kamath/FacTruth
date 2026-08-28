"""Evidence retrieval and Gemini-based evaluation for facTruth."""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
).strip()

SERPAPI_URL = "https://serpapi.com/search.json"

REQUEST_TIMEOUT_SECONDS = 15


# ---------------------------------------------------------------------------
# Credibility
# ---------------------------------------------------------------------------

HIGH_CREDIBILITY_DOMAINS = {
    "reuters.com": 0.95,
    "apnews.com": 0.95,
    "nature.com": 0.97,
    "science.org": 0.97,
    "who.int": 0.95,
    "cdc.gov": 0.95,
    "nih.gov": 0.95,
    "un.org": 0.90,
    "bbc.com": 0.90,
    "bbc.co.uk": 0.90,
    "npr.org": 0.88,
    "pubmed.ncbi.nlm.nih.gov": 0.95,
    "gov.uk": 0.85,
    "gov.in": 0.85,
}

LOW_CREDIBILITY_DOMAINS = {
    "blogspot.com": 0.25,
    "wordpress.com": 0.30,
    "medium.com": 0.40,
    "facebook.com": 0.20,
    "twitter.com": 0.25,
    "x.com": 0.25,
    "quora.com": 0.30,
    "reddit.com": 0.35,
}

DEFAULT_CREDIBILITY = 0.55

VALID_VERDICTS = {
    "True",
    "False",
    "Misleading",
    "Disputed",
    "Unverified",
}


# ---------------------------------------------------------------------------
# Language / regional search settings
# ---------------------------------------------------------------------------

LANGUAGE_SETTINGS = {
    "english": {"hl": "en", "gl": "us"},
    "tamil": {"hl": "ta", "gl": "in"},
    "telugu": {"hl": "te", "gl": "in"},
    "hindi": {"hl": "hi", "gl": "in"},
    "malayalam": {"hl": "ml", "gl": "in"},
    "kannada": {"hl": "kn", "gl": "in"},
    "bengali": {"hl": "bn", "gl": "in"},
    "marathi": {"hl": "mr", "gl": "in"},
    "gujarati": {"hl": "gu", "gl": "in"},
    "punjabi": {"hl": "pa", "gl": "in"},
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def configuration_error() -> str | None:
    """Return a user-safe configuration error."""

    missing = []

    if not SERPAPI_KEY:
        missing.append("SERPAPI_KEY")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if genai is None:
        missing.append("google-genai package")

    if missing:
        return (
            "Configuration required: set "
            + ", ".join(missing)
            + "."
        )

    return None


def _gemini_client() -> Any:
    if genai is None or not GEMINI_API_KEY:
        return None

    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Gemini JSON helpers
# ---------------------------------------------------------------------------

def _clean_json(text: str) -> dict[str, Any] | None:
    """Parse Gemini JSON, including Markdown code fences."""

    text = (text or "").strip()

    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    try:
        value = json.loads(text)

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        pass

    # Fallback: find the first JSON object
    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if not match:
        return None

    try:
        value = json.loads(match.group(0))

        if isinstance(value, dict):
            return value

    except json.JSONDecodeError:
        return None

    return None


def _generate_json(prompt: str) -> dict[str, Any] | None:
    """Ask Gemini for a JSON response."""

    client = _gemini_client()

    if client is None:
        return None

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return _clean_json(
        getattr(response, "text", "")
    )


# ---------------------------------------------------------------------------
# Source credibility
# ---------------------------------------------------------------------------

def score_domain_credibility(url: str) -> float:
    """Return an explainable 0-1 credibility score."""

    try:
        domain = (
            urlparse(url)
            .netloc
            .lower()
            .split(":")[0]
        )

        if domain.startswith("www."):
            domain = domain[4:]

    except (TypeError, ValueError):
        return DEFAULT_CREDIBILITY

    if not domain:
        return DEFAULT_CREDIBILITY

    for known, score in HIGH_CREDIBILITY_DOMAINS.items():

        if (
            domain == known
            or domain.endswith("." + known)
        ):
            return score

    if domain.endswith(".gov"):
        return 0.85

    for known, score in LOW_CREDIBILITY_DOMAINS.items():

        if (
            domain == known
            or domain.endswith("." + known)
        ):
            return score

    return DEFAULT_CREDIBILITY


# ---------------------------------------------------------------------------
# Step 1: Extract claim and search queries
# ---------------------------------------------------------------------------

def extract_claim_and_queries(
    raw_text: str,
) -> dict[str, Any]:

    prompt = f"""
You are preparing a claim for a fact-checking system.

The input may be:
- a news headline
- a social media post
- a forwarded WhatsApp message
- a statement
- a question
- text in English, Tamil, Telugu, Hindi, or another language.

Perform these tasks:

1. Detect the language.
2. Extract the main checkable factual claim.
3. Keep the original-language claim.
4. Translate that claim into English.
5. Determine whether the claim is time-sensitive.
6. Generate THREE concise English search queries.
7. Generate TWO concise search queries in the original language.

IMPORTANT:
- Do not invent facts.
- If the input is already English, native queries can also be English.
- If the input is a news headline, preserve important names, locations,
  dates, organizations, and numbers.

Input:

\"\"\"
{raw_text}
\"\"\"

Return ONLY valid JSON:

{{
    "language": "detected language",
    "claim_original": "one neutral checkable claim",
    "claim_english": "English translation",
    "time_sensitive": true,
    "queries_english": [
        "query 1",
        "query 2",
        "query 3"
    ],
    "queries_native": [
        "query 1",
        "query 2"
    ]
}}
"""

    parsed = _generate_json(prompt)

    fallback = {
        "language": "English",
        "claim_original": raw_text,
        "claim_english": raw_text,
        "time_sensitive": True,
        "queries_english": [raw_text],
        "queries_native": [],
    }

    if not parsed:
        return fallback

    result = {
        **fallback,
        **parsed,
    }

    for key in (
        "queries_english",
        "queries_native",
    ):

        value = result.get(key)

        if isinstance(value, list):

            result[key] = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        else:
            result[key] = []

    if not result["queries_english"]:
        result["queries_english"] = [raw_text]

    return result


# ---------------------------------------------------------------------------
# Step 2: SerpApi
# ---------------------------------------------------------------------------

def serpapi_search(
    query: str,
    engine: str = "google",
    num: int = 5,
    hl: str = "en",
    gl: str = "us",
) -> list[dict[str, str]]:

    """Run one SerpApi search and normalize the results."""

    if not SERPAPI_KEY:
        print("[SerpApi] ERROR: SERPAPI_KEY is missing.")
        return []

    if not query.strip():
        return []

    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": engine,
        "num": num,
        "hl": hl,
        "gl": gl,
    }

    try:

        response = requests.get(
            SERPAPI_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        print(
            f"[SerpApi] {engine} | "
            f"HTTP {response.status_code} | "
            f"{query}"
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as exc:

        print(
            f"[SerpApi] REQUEST ERROR: {exc}"
        )

        return []

    except ValueError as exc:

        print(
            f"[SerpApi] JSON ERROR: {exc}"
        )

        return []

    if not isinstance(data, dict):
        print("[SerpApi] Invalid response format.")
        return []

    if data.get("error"):

        print(
            f"[SerpApi] API ERROR: "
            f"{data.get('error')}"
        )

        return []

    result_key = {
        "google": "organic_results",
        "google_news": "news_results",
        "google_scholar": "organic_results",
        "bing": "organic_results",
    }.get(
        engine,
        "organic_results",
    )

    raw_results = data.get(
        result_key,
        [],
    )

    if not isinstance(raw_results, list):

        print(
            f"[SerpApi] No {result_key} "
            f"returned for query: {query}"
        )

        return []

    normalized = []

    for item in raw_results[:num]:

        if not isinstance(item, dict):
            continue

        link = str(
            item.get("link")
            or item.get("redirect_link")
            or ""
        ).strip()

        if not link:
            continue

        publication = item.get(
            "publication_info"
        )

        if not isinstance(
            publication,
            dict,
        ):
            publication = {}

        normalized.append(
            {
                "title": str(
                    item.get("title")
                    or "Untitled source"
                ),
                "snippet": str(
                    item.get("snippet")
                    or item.get("summary")
                    or ""
                ),
                "link": link,
                "date": str(
                    item.get("date")
                    or publication.get("summary")
                    or ""
                ),
                "engine": engine,
            }
        )

    print(
        f"[SerpApi] {engine}: "
        f"{len(normalized)} usable results"
    )

    return normalized


# ---------------------------------------------------------------------------
# Step 3: Multi-engine search
# ---------------------------------------------------------------------------

def multi_engine_search(
    queries: list[str],
    language: str = "English",
) -> list[dict[str, str]]:

    """Search Google + Google News and deduplicate results."""

    settings = LANGUAGE_SETTINGS.get(
        language.lower(),
        {
            "hl": "en",
            "gl": "us",
        },
    )

    hl = settings["hl"]
    gl = settings["gl"]

    results = []
    seen = set()

    clean_queries = list(
        dict.fromkeys(
            q.strip()
            for q in queries
            if isinstance(q, str)
            and q.strip()
        )
    )

    # Google + Google News are enough for this news-focused MVP.
    engines = (
        "google",
        "google_news",
    )

    for query in clean_queries:

        for engine in engines:

            items = serpapi_search(
                query,
                engine=engine,
                num=5,
                hl=hl,
                gl=gl,
            )

            for item in items:

                canonical = (
                    item["link"]
                    .rstrip("/")
                    .lower()
                )

                if canonical in seen:
                    continue

                seen.add(canonical)
                results.append(item)

    return results


# ---------------------------------------------------------------------------
# No evidence
# ---------------------------------------------------------------------------

def _no_evidence_result(
    search_error: str | None = None,
) -> dict[str, Any]:

    message = (
        "No usable live search results were "
        "retrieved for this claim."
    )

    if search_error:
        message += f" Search error: {search_error}"

    return {
        "verdict": "Unverified",
        "confidence": 0,
        "summary": message,
        "contradictions": [],
        "sources_used": [],
        "num_sources_checked": 0,
    }


# ---------------------------------------------------------------------------
# Step 4: Gemini evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    claim: str,
    language: str,
    time_sensitive: bool,
    sources: list[dict[str, Any]],
) -> dict[str, Any] | None:

    evidence = json.dumps(
        sources[:10],
        ensure_ascii=False,
        indent=2,
    )

    prompt = f"""
You are a careful fact-checking editor.

Evaluate the claim ONLY from the supplied search-result evidence.

Search snippets are evidence leads, not absolute proof.

Do not invent information.

Do not use outside knowledge.

Do not automatically trust a source simply because its domain is reputable.

Compare the claim against the actual evidence.

Claim:
{claim}

Language:
{language}

Time-sensitive:
{time_sensitive}

Evidence:
{evidence}

Return ONLY valid JSON:

{{
    "verdict": "True|False|Misleading|Disputed|Unverified",
    "confidence": 0,
    "summary": "brief evidence-grounded explanation",
    "contradictions": [
        "specific disagreement or limitation"
    ]
}}

Rules:

- True = evidence supports the claim.
- False = strong evidence contradicts the claim.
- Misleading = partly true but missing important context.
- Disputed = credible sources materially disagree.
- Unverified = insufficient evidence.

Confidence must be an integer from 0 to 100.

Write the summary in the input language where practical.
"""

    parsed = _generate_json(prompt)

    if not parsed:
        return None

    verdict = str(
        parsed.get(
            "verdict",
            "Unverified",
        )
    ).strip().title()

    if verdict not in VALID_VERDICTS:
        verdict = "Unverified"

    try:

        confidence = int(
            float(
                parsed.get(
                    "confidence",
                    0,
                )
            )
        )

        confidence = max(
            0,
            min(100, confidence),
        )

    except (
        TypeError,
        ValueError,
    ):
        confidence = 0

    contradictions = parsed.get(
        "contradictions",
        [],
    )

    if not isinstance(
        contradictions,
        list,
    ):
        contradictions = []

    return {
        "verdict": verdict,
        "confidence": confidence,
        "summary": str(
            parsed.get("summary")
            or "The available evidence could not support a clear conclusion."
        ),
        "contradictions": [
            str(item)
            for item in contradictions[:5]
            if str(item).strip()
        ],
    }


# ---------------------------------------------------------------------------
# Main fact-check pipeline
# ---------------------------------------------------------------------------

def run_fact_check(
    raw_text: str,
) -> dict[str, Any]:

    """Run the complete fact-checking pipeline."""

    raw_text = (
        raw_text
        or ""
    ).strip()

    if not raw_text:

        return {
            "error":
            "Provide a claim, headline, or forwarded message."
        }

    if len(raw_text) > 8000:

        return {
            "error":
            "The submitted text is too long. "
            "Please keep it under 8,000 characters."
        }

    error = configuration_error()

    if error:
        return {
            "error": error
        }

    # ---------------------------------------------------------------
    # Gemini claim preparation
    # ---------------------------------------------------------------

    try:

        extracted = (
            extract_claim_and_queries(
                raw_text
            )
        )

    except Exception as exc:

        print(
            "[Gemini extraction ERROR]",
            repr(exc)
        )

        return {
            "error":
            "Gemini could not prepare the claim "
            "for searching. Please try again."
        }

    # ---------------------------------------------------------------
    # Build search queries
    # ---------------------------------------------------------------

    queries = (
        extracted.get(
            "queries_english",
            []
        )
        +
        extracted.get(
            "queries_native",
            []
        )
    )

    # Always search the exact user input too.
    if raw_text not in queries:
        queries.append(raw_text)

    # Very useful for pasted news headlines.
    language = str(
        extracted.get(
            "language",
            "English",
        )
    )

    if language.lower() == "english":

        queries.append(
            f'site:reuters.com "{raw_text}"'
        )

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------

    try:

        sources = multi_engine_search(
            queries,
            language,
        )

    except Exception as exc:

        print(
            "[Search ERROR]",
            repr(exc)
        )

        return {
            "error":
            "The live web search failed. "
            "Check your SerpApi key and quota."
        }

    if not sources:

        return _no_evidence_result()

    # ---------------------------------------------------------------
    # Score sources
    # ---------------------------------------------------------------

    scored = []

    for item in sources:

        scored.append(
            {
                **item,
                "credibility_score":
                score_domain_credibility(
                    item["link"]
                ),
            }
        )

    scored.sort(
        key=lambda item:
        item["credibility_score"],
        reverse=True,
    )

    # ---------------------------------------------------------------
    # Gemini evaluation
    # ---------------------------------------------------------------

    try:

        evaluation = _evaluate(
            str(
                extracted.get(
                    "claim_original",
                    raw_text,
                )
            ),
            language,
            bool(
                extracted.get(
                    "time_sensitive"
                )
            ),
            scored,
        )

    except Exception as exc:

        print(
            "[Gemini evaluation ERROR]",
            repr(exc)
        )

        return {
            "error":
            "Gemini could not evaluate "
            "the retrieved evidence."
        }

    if not evaluation:

        return {
            "error":
            "Gemini returned an unreadable evaluation."
        }

    evaluation["sources_used"] = scored[:10]

    evaluation["num_sources_checked"] = len(
        scored
    )

    evaluation["language"] = language

    evaluation["claim_original"] = str(
        extracted.get(
            "claim_original",
            raw_text,
        )
    )

    evaluation["claim_english"] = str(
        extracted.get(
            "claim_english",
            raw_text,
        )
    )

    return evaluation