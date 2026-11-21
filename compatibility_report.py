"""
Probe real job-posting URLs and write model/website_compatibility.csv.

Every URL below was taken from the site itself (search result / page link), not
invented. The script records what the extractor really did, so blocked or
JavaScript-only sites are documented instead of hidden.

Run:  python compatibility_report.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import joblib
import requests

import job_extractor as je

MODEL_PATH = Path("model/fake_job_mlp.joblib")
OUTPUT_PATH = Path("model/website_compatibility.csv")

# (platform, url, category if the download itself fails)
CASES = [
    ("Python.org", "https://www.python.org/jobs/8219/", "job board (server-rendered)"),
    ("Greenhouse", "https://job-boards.greenhouse.io/discord/jobs/8299493002", "applicant tracking system"),
    ("Lever", "https://jobs.lever.co/leverdemo/ad208490-4052-4f91-a57e-433f2d1e484b", "applicant tracking system"),
    ("Ashby", "https://jobs.ashbyhq.com/openai/8fb1615c-34bf-47c4-a1d1-b7b2f836bbd3", "applicant tracking system (JS app)"),
    ("SmartRecruiters", "https://jobs.smartrecruiters.com/ServiceNow/744000162380039-software-engineer", "applicant tracking system"),
    (
        "micro1",
        "https://jobs.micro1.ai/post/b9160cd0-6896-4788-9935-596c225594b7"
        "?utm_source=Google&utm_medium=organic&ref_site=google.com",
        "ATS hosted on Next.js",
    ),
    (
        "Internshala",
        "https://internshala.com/internship/detail/"
        "video-editing-making-internship-in-lucknow-at-lucknow-trading-house1794862117",
        "Indian job board",
    ),
    ("Wellfound", "https://wellfound.com/jobs/4427735-senior-engineer", "startup job board"),
    ("Cutshort", "https://cutshort.io/job/Senior-Full-stack-Engineer-Sprinto-Rqw1mekJ", "Indian job board"),
    (
        "Shine",
        "https://www.shine.com/jobs/net-fullstack-developer-latest/aqilea/19963710"
        "?jobtype=cr&position_in_results=1",
        "Indian job board",
    ),
    ("Apna", "https://apna.co/job/bengaluru-bangalore/solar-design-engineer-bangalore-51243098", "Indian job board"),
    (
        "Freshersworld",
        "https://www.freshersworld.com/jobs/"
        "software-engineer-jobs-opening-in-a-client-of-freshersworld-at-dhanbad-2945941",
        "Indian job board",
    ),
    ("Hirist", "https://www.hirist.tech/j/senior-software-engineer-1651311", "Indian job board (JS app)"),
    (
        "Naukri",
        "https://www.naukri.com/job-listings-software-engineer-quess-it-solutions-"
        "bengaluru-4-to-7-years-271026028458",
        "Indian job board",
    ),
    ("Indeed India", "https://in.indeed.com/viewjob?from=serp&jk=118f2fa550841877&vjs=3", "aggregator"),
    ("Glassdoor India", "https://www.glassdoor.co.in/job-listing/jv?jl=1010031969299", "aggregator"),
    ("LinkedIn", "https://www.linkedin.com/jobs/view/4471319619/", "social job board"),
    ("Foundit", "https://www.foundit.in/job/engineering-l2-bengaluru-associate-software-engineering-"
                "goldman-sachs-bengaluru-bangalore-49040983", "Indian job board"),
    (
        "Workday",
        "https://disney.wd5.myworkdayjobs.com/en-US/disneycareerdc/job/Software-Engineer_10159530-1",
        "applicant tracking system (JS app)",
    ),
]

# Cases that must always be refused, checked by the same code path.
CONTROL_CASES = [
    ("Control: invalid scheme", "ftp://example.com/job/1"),
    ("Control: localhost", "http://localhost:8501/"),
    ("Control: private IP", "http://192.168.1.10/job/1"),
    ("Control: broken host", "https://this-host-does-not-exist-12345.example/job/1"),
    ("Control: homepage", "https://example.com/"),
    ("Control: listing page", "https://www.python.org/jobs/"),
    ("Control: missing posting", "https://www.python.org/jobs/0000/"),
    ("Control: not a URL", "not a url"),
]

FIELD_ORDER = [
    "description",
    "requirements",
    "company_profile",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
]


def http_status(url: str) -> str:
    """Plain GET status code, so the CSV shows what the site answered."""
    try:
        response = requests.get(url, headers=je.HEADERS, timeout=25, allow_redirects=True)
        return str(response.status_code)
    except requests.RequestException:
        return "no response"


def has_value(value) -> bool:
    return str(value).strip() not in ("", "0")


def classify_failure(message: str) -> str:
    """Turn the extractor message into the short category used in the CSV."""
    lowered = message.lower()
    if "blocks automated" in lowered:
        return "Blocked"
    if "needs javascript" in lowered:
        return "Requires JavaScript"
    if "looks like a list of jobs" in lowered:
        return "Listing page"
    if (
        "must start with http" in lowered
        or "local addresses" in lowered
        or "private or local" in lowered
    ):
        return "Invalid or unsafe URL"
    if "job description could not be found" in lowered:
        return "Unsupported page structure"
    if "could not be found" in lowered or "not found (http" in lowered:
        return "Invalid or unreachable URL"
    return "Extraction failed"


def run_case(model, platform: str, url: str, probe_http: bool = True) -> dict:
    result = je.extract_job(url)
    row = {
        "Platform": platform,
        "Test URL": url,
        "HTTP status": http_status(url) if probe_http else "not fetched (rejected locally)",
        "Extraction success": "Yes" if result.ok else "No",
        "Extraction method": result.used if result.ok else "-",
        "Description word count": "",
        "Extracted fields": "",
        "Missing fields": "",
        "Prediction generated": "No",
        "Failure reason": "",
    }
    if not result.ok:
        row["Failure reason"] = f"{classify_failure(result.message)}: {result.message}"
        return row

    fields = result.fields
    row["Description word count"] = len(str(fields.get("description") or "").split())
    extracted = [name for name in FIELD_ORDER if has_value(fields.get(name))]
    missing = [name for name in FIELD_ORDER if not has_value(fields.get(name))]
    row["Extracted fields"] = ", ".join(extracted) if extracted else "none"
    row["Missing fields"] = ", ".join(missing) if missing else "none"

    if model is not None:
        from app import predict_fields  # reuse the exact app prediction path

        prediction = predict_fields(model, fields)
        label = "suspicious" if prediction["is_fraud"] else "genuine"
        row["Prediction generated"] = f"Yes ({label}, probability {prediction['probability']:.3f})"
    return row


def main() -> None:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is None:
        print(f"! {MODEL_PATH} not found - predictions will be skipped")

    rows = [run_case(model, platform, url) for platform, url, _ in CASES]
    rows += [run_case(model, name, url, probe_http=False) for name, url in CONTROL_CASES]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_PATH} ({len(rows)} rows)\n")
    for row in rows:
        state = "OK  " if row["Extraction success"] == "Yes" else "FAIL"
        detail = row["Extraction method"] if state == "OK  " else row["Failure reason"].split(":")[0]
        print(f"{state} {row['Platform']:26s} {row['HTTP status']:>6s}  {detail}")


if __name__ == "__main__":
    main()
