"""
job_extractor.py
----------------
Fetch a public job posting web page and pull out the fields our model needs.

Two extraction strategies, in order:
  1. schema.org JSON-LD  (<script type="application/ld+json"> with "@type": "JobPosting")
     -> used by most company career sites, it is the reliable path.
  2. Plain HTML heuristics (description containers, headings, meta tags)
     -> best effort for sites without JSON-LD.

The extracted values are mapped onto the exact categories seen during training
(e.g. "FULL_TIME" -> "Full-time", "bachelor" -> "Bachelor's Degree").
Anything that cannot be found is left as an empty string, which the existing
preprocessing pipeline already handles as a missing value.

This module never tries to bypass a login, CAPTCHA, paywall or bot protection:
blocked pages simply produce a friendly error and the app falls back to manual input.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------- settings ----
REQUEST_TIMEOUT = 12          # seconds
MAX_BYTES = 3_000_000         # do not download giant pages
MAX_URL_LENGTH = 2048         # longer strings are never a real posting address
MIN_DESCRIPTION_WORDS = 40    # below this we consider the extraction failed

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 StudentProjectBot/1.0"
    ),
    "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
    "Accept-Language": "en-US,en;q=0.8",
}

# Field names must match preprocessing.build_input_dataframe exactly.
FIELD_NAMES = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "description",
    "requirements",
    "company_profile",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "salary_range",
]


@dataclass
class ExtractionResult:
    """Everything the app needs to know about one URL fetch."""

    ok: bool
    fields: dict
    message: str = ""
    url: str = ""
    page_title: str = ""
    site: str = ""
    used: str = ""                      # "JSON-LD" or "HTML"
    warnings: list = field(default_factory=list)


# ------------------------------------------------------------------ helpers ----
def empty_fields() -> dict:
    return {name: "" for name in FIELD_NAMES}


def _clean_text(text: Any) -> str:
    """Collapse whitespace and drop control / non-ASCII junk."""
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[^\x09\x0a\x20-\x7e]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def html_fragment_to_text(fragment: str) -> str:
    """Turn an HTML snippet (JSON-LD descriptions are HTML) into plain text."""
    if not fragment:
        return ""
    soup = BeautifulSoup(fragment, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for block in soup.find_all(["p", "li", "div", "h1", "h2", "h3", "h4", "tr"]):
        block.append("\n")
    return _clean_text(soup.get_text("\n"))


def word_count(text: str) -> int:
    return len(text.split())


def _walk_json(node: Any):
    """Yield every dict inside a (possibly nested) JSON structure."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_json(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_json(item)


def _as_text(value: Any) -> str:
    """JSON-LD values are sometimes a string, a list or a dict."""
    if value is None:
        return ""
    if isinstance(value, str):
        return _clean_text(html_fragment_to_text(value))
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return _clean_text("\n".join(p for p in parts if p))
    if isinstance(value, dict):
        for key in ("name", "description", "value", "text"):
            if key in value:
                return _as_text(value[key])
    return ""


# -------------------------------------------------------------------- URLs ----
def check_url(url: str) -> str:
    """Return an error message, or '' when the URL is acceptable to fetch."""
    url = (url or "").strip()
    if not url:
        return "Please paste a job posting URL first."
    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        return "The URL must start with http:// or https:// (for example https://www.example.com/jobs/123)."

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return "That URL does not contain a valid website address."
    if len(url) > MAX_URL_LENGTH:
        return "That URL is too long. Please paste the address of one job posting."

    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return "Local addresses cannot be fetched from the web. Please use a public job posting URL."

    # Block private / loopback / link-local IPs (basic safety guard).
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return "Private or local network addresses are not supported. Please use a public URL."
    except socket.gaierror:
        return f"The website '{host}' could not be found. Please check the URL for typos."
    except ValueError:
        pass
    return ""


def fetch_page(url: str) -> tuple[BeautifulSoup | None, str]:
    """
    Download the page and return (soup, error_message).
    Exactly one of the two is ever non-empty.
    """
    try:
        response = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True
        )
    except requests.exceptions.Timeout:
        return None, f"The website did not respond within {REQUEST_TIMEOUT} seconds (timeout)."
    except requests.exceptions.SSLError:
        return None, "The website has an invalid or expired security certificate, so it was not opened."
    except requests.exceptions.TooManyRedirects:
        return None, "The website kept redirecting in a loop, so the page could not be loaded."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the website. It may be offline, or the address may be wrong."
    except requests.exceptions.RequestException as exc:                     # any other requests error
        return None, f"The page could not be downloaded ({type(exc).__name__})."

    status = response.status_code

    # Sites that refuse automated visitors; we do not try to get around this.
    if status in (401, 403, 429, 999):
        return None, (
            f"This website blocks automated visitors (HTTP {status}). "
            "It probably requires a login, a CAPTCHA or a paywall, which this project does not bypass."
        )
    if status in (404, 410):
        return None, f"The page was not found (HTTP {status}). The job posting may have been removed."
    if status >= 400:
        return None, f"The website returned an error (HTTP {status})."

    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and "html" not in content_type and "xml" not in content_type:
        return None, f"This URL does not point to a web page (content type: {content_type})."

    chunks, total = [], 0
    for chunk in response.iter_content(chunk_size=65536):
        chunks.append(chunk)
        total += len(chunk)
        if total > MAX_BYTES:
            break

    encoding = response.encoding or "utf-8"
    try:
        html = b"".join(chunks).decode(encoding, errors="replace")
    except (LookupError, UnicodeDecodeError):
        html = b"".join(chunks).decode("utf-8", errors="replace")

    soup = BeautifulSoup(html, "html.parser")
    return soup, ""


# ------------------------------------------------------------ JSON-LD path ----
def _jobposting_nodes(data: Any) -> list:
    """Every schema.org JobPosting object inside a parsed JSON structure."""
    found = []
    for node in _walk_json(data):
        node_type = node.get("@type")
        types = node_type if isinstance(node_type, list) else [node_type]
        if any(isinstance(t, str) and t.lower() == "jobposting" for t in types):
            found.append(node)
    return found


def _richest_jobposting(candidates: list) -> dict | None:
    """Several JobPosting blocks can exist (e.g. a stub + a full one); keep the fullest."""
    if not candidates:
        return None

    def richness(node: dict) -> int:
        text = str(node.get("description") or "")
        return len(text)

    return max(candidates, key=richness)


def find_jobposting_jsonld(soup: BeautifulSoup, allow_embedded: bool = True) -> dict | None:
    """
    Return the schema.org JobPosting object for the page, if there is one.

    Two places are searched:
      1. real `<script type="application/ld+json">` blocks (the standard case)
      2. JobPosting JSON embedded inside other inline scripts, which is how
         Next.js/React Server Component pages ship their data
         (`self.__next_f.push([1,"..."])`, `__NEXT_DATA__`, ...).
         The payload is unescaped as a JS string literal and the enclosing JSON
         object is decoded with json.JSONDecoder - nothing is evaluated.
    """
    candidates = []

    for script in soup.find_all("script"):
        script_type = (script.get("type") or "").lower()
        if "ld+json" not in script_type:
            continue
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        candidates.extend(_jobposting_nodes(data))

    if not candidates and allow_embedded:
        for payload in _script_payload_strings(soup):
            node = _find_jobposting_by_decoding(payload)
            if node is not None:
                candidates.append(node)

    return _richest_jobposting(candidates)


def _script_payload_strings(soup: BeautifulSoup, budget: int = 6_000_000):
    """
    Yield the text of inline scripts, plus the unescaped JS string literals
    inside them (that is where Next.js streaming payloads keep their JSON).
    """
    for script in soup.find_all("script"):
        if script.get("src"):
            continue
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        budget -= len(raw)
        if budget < 0:
            break
        yield raw
        if "JobPosting" not in raw and "@type" not in raw:
            continue
        # JS string literals long enough to hold a job description.
        for literal in re.findall(r'"((?:[^"\\]|\\.){60,})"', raw):
            if "JobPosting" not in literal and "@type" not in literal:
                continue
            try:
                yield json.loads('"' + literal + '"')
            except (json.JSONDecodeError, ValueError):
                continue


def _find_jobposting_by_decoding(text: str) -> dict | None:
    """
    Find a JobPosting object inside a plain text blob by decoding the JSON
    object that encloses its "@type" key (balanced brace matching via
    json.JSONDecoder.raw_decode - safe, no eval).
    """
    if "JobPosting" not in text:
        return None

    decoder = json.JSONDecoder()
    for match in list(re.finditer(r'"@type"', text))[:120]:
        index = text.rfind("{", 0, match.start())
        attempts = 0
        while index != -1 and attempts < 25:
            try:
                obj, _end = decoder.raw_decode(text, index)
            except (json.JSONDecodeError, ValueError):
                index = text.rfind("{", 0, index)
                attempts += 1
                continue
            nodes = _jobposting_nodes(obj)
            if nodes:
                return _richest_jobposting(nodes)
            break
    return None


def _salary_from_jsonld(jobposting: dict) -> str:
    """Build a 'min-max' string like the dataset uses, e.g. '50000-80000'."""
    blocks = [jobposting.get("baseSalary"), jobposting.get("estimatedSalary")]
    for block in blocks:
        for node in _walk_json(block):
            value = node.get("value")
            if isinstance(value, dict):
                low = value.get("minValue") or value.get("value")
                high = value.get("maxValue") or value.get("value")
                numbers = []
                for number in (low, high):
                    try:
                        numbers.append(int(float(number)))
                    except (TypeError, ValueError):
                        pass
                if numbers:
                    return f"{min(numbers)}-{max(numbers)}"
            elif isinstance(value, (int, float)) or (
                isinstance(value, str) and value.replace(",", "").isdigit()
            ):
                number = int(float(str(value).replace(",", "")))
                return f"{number}-{number}"
    return ""


def fields_from_jsonld(jobposting: dict) -> dict:
    fields = empty_fields()

    fields["description"] = _clean_text(
        html_fragment_to_text(_as_text(jobposting.get("description")))
    )

    organisation = jobposting.get("hiringOrganization") or {}
    if isinstance(organisation, list):
        organisation = organisation[0] if organisation else {}
    if not isinstance(organisation, dict):
        organisation = {}
    fields["company_profile"] = _clean_text(
        html_fragment_to_text(_as_text(organisation.get("description")))
    )

    # Requirements can be spread over several schema.org properties.
    requirement_parts = []
    for key in (
        "qualifications",
        "experienceRequirements",
        "educationRequirements",
        "responsibilities",
        "skills",
        "jobBenefits",
    ):
        text = _as_text(jobposting.get(key))
        if text:
            requirement_parts.append(text)
    fields["requirements"] = _clean_text("\n".join(requirement_parts))

    fields["salary_range"] = _salary_from_jsonld(jobposting)

    employment = jobposting.get("employmentType")
    employment_values = employment if isinstance(employment, list) else [employment]
    for value in employment_values:
        mapped = map_employment_type(str(value or ""))
        if mapped:
            fields["employment_type"] = mapped
            break

    fields["industry"] = _as_text(jobposting.get("industry"))
    fields["function"] = _as_text(
        jobposting.get("occupationalCategory") or jobposting.get("jobFunction")
    )

    if str(jobposting.get("jobLocationType") or "").upper() == "TELECOMMUTE":
        fields["telecommuting"] = 1
    logo = organisation.get("logo") or jobposting.get("logo")
    if logo:
        fields["has_company_logo"] = 1

    return {key: value for key, value in fields.items() if value != ""}


# -------------------------------------------------------------- HTML path ----
JOB_BODY_WORDS = re.compile(
    r"(responsibilit\w*|qualification\w*|requirement\w*|what you.ll do|"
    r"about the (?:role|job|position|team)|job description|who you are|"
    r"we are looking for|we.re looking for|you will|the role|your role|"
    r"what you.ll be doing|day[- ]to[- ]day)",
    re.I,
)
DETAIL_LINK_RE = re.compile(r"(/jobs?/|/job-|/remote-jobs?/|/careers?/|/positions?/)", re.I)
DESCRIPTION_HINTS = (
    "job-description", "jobdescription", "job_description", "job-details", "jobdetails",
    "job-content", "jobcontent", "posting", "description", "job-desc", "role-description",
)
NOISE_TAGS = ("script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg")


def _visible_text(tag) -> str:
    """Text of a tag with scripts, menus, forms and footers removed."""
    clone = BeautifulSoup(str(tag), "html.parser")
    for noise in clone(list(NOISE_TAGS)):
        noise.decompose()
    for block in clone.find_all(["p", "li", "div", "h1", "h2", "h3", "h4", "br", "tr"]):
        block.append("\n")
    return _clean_text(clone.get_text("\n"))


def guess_description_html(soup: BeautifulSoup) -> str:
    """Best-effort description: likely containers first, then the largest block."""
    best_text, best_score = "", 0
    for tag in soup.find_all(["article", "main", "section", "div"]):
        identifier = " ".join(
            [tag.get("id") or "", " ".join(tag.get("class") or []), tag.get("itemprop") or ""]
        ).lower()
        text = _visible_text(tag)
        words = word_count(text)
        if words < 30:
            continue

        score = min(words, 1500)
        if any(hint in identifier for hint in DESCRIPTION_HINTS):
            score += 250
        if "jobdescription" in identifier.replace("-", "").replace("_", ""):
            score += 250
        if JOB_BODY_WORDS.search(text):
            score += 150
        if re.search(r"job|career|position|role", identifier):
            score += 100
        # A big wrapper that mostly holds navigation links is the page, not the
        # posting (e.g. Freshersworld), so take links into account.
        score -= 6 * len(tag.find_all("a"))

        if score > best_score:
            best_text, best_score = text, score

    if best_text:
        return best_text

    for selector in ('meta[property="og:description"]', 'meta[name="description"]'):
        meta = soup.select_one(selector)
        if meta and word_count(_clean_text(meta.get("content")) or "") >= MIN_DESCRIPTION_WORDS:
            return _clean_text(meta.get("content"))

    main = soup.find("main") or soup.find("article") or soup.body
    return _visible_text(main) if main else ""


SECTION_HEADINGS = re.compile(
    r"^(requirements?|qualifications?|required(?: skills)?|skills?|must have[s]?|"
    r"what you(?:'ll| will)? need|who you are|your profile|about you|education)\b[^\n]{0,60}$",
    re.I,
)
OTHER_HEADINGS = re.compile(
    r"^(responsibilit\w*|what you(?:'ll| will)? do|about (?:us|the company|the team|our company)|"
    r"benefits?|perks?|why (?:join|work)|the role|company (?:overview|profile)|how to apply|"
    r"compensation|salary|apply now)\b[^\n]{0,60}$",
    re.I,
)
# Sections that mean "the posting text has ended" - stop collecting there.
STOP_HEADINGS = re.compile(
    r"^(job particulars|related jobs?|similar jobs?|more jobs?|other jobs?|"
    r"recommended jobs?|browse (?:jobs|all)|recently posted|latest jobs?|"
    r"share this job|report (?:this )?job|applied)\b[^\n]{0,40}$",
    re.I,
)

# A requirements / company-profile block longer than this is almost always the
# splitter leaking into the rest of the page (related jobs, footer, ...), so it
# is trimmed to keep the feature values realistic.
MAX_SECTION_WORDS = 400


def split_sections(text: str) -> tuple[str, str, str]:
    """
    Split plain posting text into (description, requirements, company_profile).
    Word counts are what the model uses, so approximate sections are enough.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return "", "", ""

    requirements, profile = [], []
    current = None
    body = []
    for line in lines:
        if STOP_HEADINGS.match(line):
            break
        if SECTION_HEADINGS.match(line):
            current = "requirements"
            continue
        if OTHER_HEADINGS.match(line):
            lowered = line.lower()
            current = "profile" if "company" in lowered or "about us" in lowered else None
            continue
        if len(line.split()) <= 8 and line.endswith(":"):
            current = None
        if current == "requirements":
            requirements.append(line)
        elif current == "profile":
            profile.append(line)
        else:
            body.append(line)

    description = _clean_text("\n".join(body)) or _clean_text(text)
    return description, _trim_section("\n".join(requirements)), _trim_section("\n".join(profile))


def _trim_section(text: str, max_words: int = MAX_SECTION_WORDS) -> str:
    """Keep a section at a realistic size so one heading cannot swallow the page."""
    cleaned = _clean_text(text)
    words = cleaned.split()
    if len(words) <= max_words:
        return cleaned
    return " ".join(words[:max_words])


def fields_from_html(soup: BeautifulSoup, url: str) -> dict:
    fields = empty_fields()
    description_text = guess_description_html(soup)
    description, requirements, profile = split_sections(description_text)

    fields["description"] = description
    fields["requirements"] = requirements
    fields["company_profile"] = profile

    # Company name from common meta tags -> short company profile.
    site_name = ""
    for selector in ('meta[property="og:site_name"]', 'meta[name="application-name"]'):
        meta = soup.select_one(selector)
        if meta and (meta.get("content") or "").strip():
            site_name = _clean_text(meta.get("content"))
            break
    if not fields["company_profile"] and site_name:
        fields["company_profile"] = site_name

    # Salary range written as text, e.g. "$50,000 - $70,000".
    salary_match = re.search(
        r"\$\s?(\d[\d,]{2,})\s*(?:-|to)\s*\$?\s?(\d[\d,]{2,})", _clean_text(soup.get_text(" "))
    )
    if salary_match:
        low = int(salary_match.group(1).replace(",", ""))
        high = int(salary_match.group(2).replace(",", ""))
        fields["salary_range"] = f"{min(low, high)}-{max(low, high)}"

    # Company logo: an <img> that looks like a logo in the page header.
    for image in soup.find_all("img"):
        marker = " ".join(
            [
                (image.get("src") or ""),
                (image.get("alt") or ""),
                (image.get("class") and " ".join(image.get("class")) or ""),
                (image.get("id") or ""),
            ]
        ).lower()
        if "logo" in marker:
            fields["has_company_logo"] = 1
            break
    if not fields["has_company_logo"]:
        for meta in soup.select('meta[property="og:image"]'):
            if "logo" in (meta.get("content") or "").lower():
                fields["has_company_logo"] = 1
                break

    # Application questions: a real form with radio buttons / checkboxes.
    question_inputs = soup.select(
        "form input[type=radio], form input[type=checkbox], form select, "
        '[class*="question"] input, [id*="question"] input'
    )
    page_text = _clean_text(soup.get_text(" "))
    if len(question_inputs) >= 2 or re.search(r"screening questions?|application questions?", page_text, re.I):
        fields["has_questions"] = 1

    # Telecommuting / remote work.
    if re.search(
        r"\b(fully remote|remote (?:position|role|job|work)|work from home|work-from-home|"
        r"telecommut\w*|remote-first)\b",
        f"{_page_title(soup)}\n{fields['description'][:2000]}",
        re.I,
    ):
        fields["telecommuting"] = 1

    return {key: value for key, value in fields.items() if value != ""}


# ------------------------------------------------------- category mappings ----
EMPLOYMENT_MAP = {
    "full_time": "Full-time", "full-time": "Full-time", "fulltime": "Full-time",
    "part_time": "Part-time", "part-time": "Part-time", "parttime": "Part-time",
    "contractor": "Contract", "contract": "Contract", "freelance": "Contract",
    "temporary": "Temporary", "temp": "Temporary", "seasonal": "Temporary",
    "intern": "Other", "internship": "Other", "volunteer": "Other",
    "per_diem": "Other", "other": "Other",
}

EXPERIENCE_RULES = [
    (r"\bintern(?:ship)?\b", "Internship"),
    (r"\b(entry[- ]level|junior|jr\.?|graduate|fresh graduate)\b", "Entry level"),
    (r"\bassociate\b", "Associate"),
    (r"\b(director|head of)\b", "Director"),
    (r"\b(vice president|vp|chief|cto|ceo|cfo|executive)\b", "Executive"),
    (r"\b(mid[- ]senior|mid[- ]level|senior|lead|principal|staff)\b", "Mid-Senior level"),
]

# When only "N years of experience" is stated, use the years as the seniority hint.
YEARS_RULES = [
    (r"\b(?:[5-9]|1\d|2\d)\s*(?:\+|or more|plus)?\s*years?\b", "Mid-Senior level"),
    (r"\b(?:five|six|seven|eight|nine|ten|more than five)\s*(?:or more|\+)?\s*years?\b", "Mid-Senior level"),
    (r"\b(?:[2-4]|two|three|four)\s*(?:\+|or more|plus)?\s*years?\b", "Associate"),
    (r"\b(?:freshers?|no (?:prior )?(?:work )?experience|0\s*\+?\s*years?)\b", "Entry level"),
    (r"\b(?:1|one)\s*(?:\+|or more|plus)?\s*years?\b", "Entry level"),
]

EDUCATION_RULES = [
    (r"\b(ph\.?d|doctorate|doctoral)\b", "Doctorate"),
    (r"\b(master'?s?|m\.?sc|mba|postgraduate)\b", "Master's Degree"),
    (r"\b(bachelor'?s?|b\.?sc|b\.?tech|b\.?e\.?|undergraduate degree|4[- ]year degree)\b", "Bachelor's Degree"),
    (r"\bassociate'?s? degree\b", "Associate Degree"),
    (r"\b(high school|ged|secondary school)\b", "High School or equivalent"),
    (r"\b(some college)\b", "Some College Coursework Completed"),
    (r"\b(certification|certified|vocational|diploma)\b", "Certification"),
]


def map_employment_type(value: str) -> str:
    key = re.sub(r"[^a-z_\- ]", "", (value or "").lower()).strip()
    if key in EMPLOYMENT_MAP:
        return EMPLOYMENT_MAP[key]
    key_underscore = key.replace(" ", "_").replace("-", "_")
    if key_underscore in EMPLOYMENT_MAP:
        return EMPLOYMENT_MAP[key_underscore]
    for pattern, label in (
        (r"part[- ]?time", "Part-time"),
        (r"full[- ]?time", "Full-time"),
        (r"contract", "Contract"),
        (r"temporar", "Temporary"),
        (r"intern", "Other"),
    ):
        if re.search(pattern, key):
            return label
    return ""


def map_experience(text: str) -> str:
    """
    Seniority from explicit words first ("Senior", "Director"), then from the
    number of years asked for, because most postings only state years.
    """
    explicit = map_from_rules(text, EXPERIENCE_RULES)
    if explicit:
        return explicit
    return map_from_rules(text, YEARS_RULES)


def map_from_rules(text: str, rules) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return ""
    for pattern, label in rules:
        if re.search(pattern, lowered):
            return label
    return ""


def normalize_category(value: str, allowed: list) -> str:
    """
    Map a free-text value onto one of the categories seen during training.
    Exact (case/punctuation-insensitive) match first, then a substring match.
    Returns '' when nothing sensible matches, so the pipeline treats it as missing.
    """
    value = _clean_text(value)
    if not value or not allowed:
        return ""

    def squash(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    target = squash(value)
    for option in allowed:
        if squash(option) == target:
            return option
    for option in allowed:
        squashed = squash(option)
        if squashed and (squashed in target or target in squashed) and len(target) >= 4:
            return option
    for token in target.split():
        if len(token) < 5:
            continue
        for option in allowed:
            if token in squash(option).split():
                return option
    return ""


# ------------------------------------------------------------------- public ----
def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    heading = soup.find("h1")
    return _clean_text(heading.get_text(" ")) if heading else ""


def _site_name(soup: BeautifulSoup, url: str) -> str:
    for selector in ('meta[property="og:site_name"]', 'meta[name="application-name"]'):
        meta = soup.select_one(selector)
        if meta and (meta.get("content") or "").strip():
            return _clean_text(meta.get("content"))
    return urlparse(url).hostname or ""


def _count_detail_links(soup: BeautifulSoup) -> int:
    """Count links that look like individual job postings (a listing page has many)."""
    return len(
        [
            a
            for a in soup.find_all("a", href=True)
            if DETAIL_LINK_RE.search(a["href"]) and re.search(r"\d", a["href"])
        ]
    )


def _looks_like_job_page(soup: BeautifulSoup, title: str, description: str) -> tuple[bool, str]:
    """
    Decide whether this page is ONE job posting.

    Returns (True, "") or (False, reason). Without this check a job *listing*
    page or a marketing homepage would be 'analyzed' as if it were a posting.
    """
    if _count_detail_links(soup) >= 8:
        return False, (
            "This page looks like a list of jobs rather than a single posting. "
            "Please open the job you are interested in and paste that link instead."
        )
    if not JOB_BODY_WORDS.search(description[:6000]):
        return False, (
            "This page does not contain a normal job description "
            "(no responsibilities, requirements or qualifications section)."
        )
    return True, ""


def _captcha_or_login_wall(soup: BeautifulSoup) -> str:
    """Detect a real bot wall. Checked BEFORE extraction, because nothing can be read."""
    text = _clean_text(soup.get_text(" ")).lower()
    snippet = text[:1500]
    if "captcha" in snippet or "are you a robot" in snippet or "verify you are human" in snippet:
        return "The page shows a CAPTCHA or bot check, which this project does not bypass."
    if re.search(r"\b(sign in|log in|login) to (view|see|continue)", snippet):
        return "The job posting is behind a login, which this project does not bypass."
    return ""


def _looks_javascript_only(soup: BeautifulSoup) -> str:
    """
    Only used when the extraction already FAILED: explains that the page is a
    JavaScript shell. It is never used to reject a page whose data we could read
    (many pages say 'enable JavaScript' yet still embed the posting in JSON-LD).
    """
    lowered = _clean_text(soup.get_text(" ")).lower()
    markers = ("enable javascript", "javascript to run this app", "requires javascript")
    if any(marker in lowered for marker in markers):
        return (
            "The page needs JavaScript to show the job posting, so it cannot be read by a "
            "simple fetch."
        )
    return ""


def extract_job(
    url: str,
    allowed_industries: list | None = None,
    allowed_functions: list | None = None,
) -> ExtractionResult:
    """
    Fetch a public job posting and return an ExtractionResult.

    ok=True  -> a description was found (some other fields may still be missing)
    ok=False -> nothing usable; the caller should offer manual input
    """
    url = (url or "").strip()
    error = check_url(url)
    if error:
        return ExtractionResult(ok=False, fields=empty_fields(), message=error, url=url)

    soup, error = fetch_page(url)
    if soup is None:
        return ExtractionResult(ok=False, fields=empty_fields(), message=error, url=url)

    title = _page_title(soup)
    site = _site_name(soup, url)

    blocked = _captcha_or_login_wall(soup)
    if blocked:
        return ExtractionResult(
            ok=False, fields=empty_fields(), message=blocked, url=url, page_title=title, site=site
        )

    warnings = []
    jobposting = find_jobposting_jsonld(soup)
    if jobposting:
        fields = fields_from_jsonld(jobposting)
        used = "JSON-LD (schema.org JobPosting)"

        # Some sites publish a stub JobPosting ("<title> job opportunity.").
        # In that case prefer the visible HTML text, but never lose a richer
        # structured description.
        if word_count(fields.get("description", "")) < MIN_DESCRIPTION_WORDS:
            html_fields = fields_from_html(soup, url)
            if word_count(html_fields.get("description", "")) > word_count(
                fields.get("description", "")
            ):
                fields["description"] = html_fields["description"]
                used = "HTML content (JobPosting JSON-LD had no real description)"
            for key, value in html_fields.items():
                if key != "description" and not fields.get(key):
                    fields[key] = value
    else:
        fields = fields_from_html(soup, url)
        used = "HTML content (no schema.org JobPosting found)"

    fields.setdefault("description", "")
    description = fields.get("description", "")
    if word_count(description) < MIN_DESCRIPTION_WORDS:
        js_wall = _looks_javascript_only(soup)
        return ExtractionResult(
            ok=False,
            fields=empty_fields(),
            message=js_wall
            or (
                "A job description could not be found on this page. "
                "The link may point to a list of jobs, a homepage, or a page that loads its "
                "content with JavaScript."
            ),
            url=url,
            page_title=title,
            site=site,
        )

    if not jobposting:
        looks_like_job, reason = _looks_like_job_page(soup, title, description)
        if not looks_like_job:
            return ExtractionResult(
                ok=False,
                fields=empty_fields(),
                message=_looks_javascript_only(soup) or reason,
                url=url,
                page_title=title,
                site=site,
            )

    # ---- normalise the categorical fields onto the training categories ----
    fields["employment_type"] = map_employment_type(fields.get("employment_type", ""))
    if not fields["employment_type"]:
        fields["employment_type"] = map_employment_type(f"{title} {' '.join(description.split()[:60])}")

    # Seniority: the job title is the most reliable signal, then the requirements
    # section, then the start of the description ("junior engineers" inside the
    # description must not turn a senior role into an entry-level one).
    fields["required_experience"] = (
        map_experience(title)
        or map_experience(fields.get("requirements", ""))
        or map_experience(" ".join(description.split()[:120]))
    )

    education_source = " ".join([fields.get("requirements", ""), description])
    fields["required_education"] = map_from_rules(education_source, EDUCATION_RULES)

    fields["industry"] = normalize_category(fields.get("industry", ""), allowed_industries or [])
    fields["function"] = normalize_category(fields.get("function", ""), allowed_functions or [])

    fields["telecommuting"] = int(bool(fields.get("telecommuting", 0)))
    fields["has_company_logo"] = int(bool(fields.get("has_company_logo", 0)))
    fields["has_questions"] = int(bool(fields.get("has_questions", 0)))
    fields["salary_range"] = _clean_text(fields.get("salary_range", ""))

    # ---- tell the user which fields are missing (partial extraction) ----
    labels = {
        "employment_type": "employment type",
        "required_experience": "required experience",
        "required_education": "required education",
        "industry": "industry",
        "function": "job function",
        "requirements": "requirements text",
        "company_profile": "company profile",
        "salary_range": "salary range",
    }
    for key, label in labels.items():
        if not fields.get(key):
            warnings.append(f"{label} not found on the page")
    if not fields["has_company_logo"]:
        warnings.append("company logo not detected")
    if not fields["has_questions"]:
        warnings.append("no screening questions detected")

    final_fields = empty_fields()
    for name in FIELD_NAMES:
        final_fields[name] = fields.get(name, "")
    final_fields["description"] = description

    return ExtractionResult(
        ok=True,
        fields=final_fields,
        message="Job details extracted successfully.",
        url=url,
        page_title=title,
        site=site,
        used=used,
        warnings=warnings,
    )
