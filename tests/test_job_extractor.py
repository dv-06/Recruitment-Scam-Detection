"""
Offline tests for job_extractor.py - no network access needed.

Each test loads a saved HTML fixture and replaces the downloader with a stub, so
the parsing logic is tested exactly as it runs in the app.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import job_extractor as je  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAKE_URL = "https://jobs.example.com/posting/1"


def load_fixture(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


class FixtureCase(unittest.TestCase):
    """Serves a saved fixture instead of downloading a real page."""

    def extract(self, fixture: str, url: str = FAKE_URL):
        soup = load_fixture(fixture)
        original_fetch, original_check = je.fetch_page, je.check_url
        je.check_url = lambda value: ""  # the URL itself is not what is under test
        je.fetch_page = lambda value: (soup, "")
        try:
            return je.extract_job(url)
        finally:
            je.fetch_page, je.check_url = original_fetch, original_check


class TestUrlSafety(unittest.TestCase):
    def test_rejects_non_http_scheme(self):
        self.assertIn("http://", je.check_url("ftp://example.com/job/1"))

    def test_rejects_localhost(self):
        self.assertIn("Local addresses", je.check_url("http://localhost:8501/"))

    def test_rejects_private_ip(self):
        self.assertIn("Private or local", je.check_url("http://192.168.1.10/job/1"))

    def test_rejects_oversized_url(self):
        self.assertTrue(je.check_url("https://example.com/" + "a" * 3000))

    def test_accepts_public_https(self):
        self.assertEqual(je.check_url("https://www.python.org/jobs/8219/"), "")


class TestJsonLd(FixtureCase):
    def test_standard_jsonld_posting(self):
        result = self.extract("jsonld_posting.html")
        self.assertTrue(result.ok, result.message)
        self.assertIn("JSON-LD", result.used)
        self.assertEqual(result.fields["employment_type"], "Full-time")
        self.assertEqual(result.fields["salary_range"], "1800000-2400000")
        self.assertEqual(result.fields["required_education"], "Bachelor's Degree")
        self.assertGreaterEqual(je.word_count(result.fields["description"]), 40)

    def test_embedded_nextjs_payload_is_found(self):
        """The micro1.ai case: JobPosting JSON inside the Next.js stream, no ld+json tag."""
        result = self.extract("embedded_rsc_posting.html")
        self.assertTrue(result.ok, result.message)
        self.assertIn("JSON-LD", result.used)
        self.assertIn("Robotics Data Collection", result.fields["description"])
        self.assertEqual(result.fields["telecommuting"], 1)
        self.assertEqual(result.fields["employment_type"], "Contract")

    def test_stub_jsonld_description_is_replaced_by_html(self):
        """The apna.co case: JSON-LD exists but its description is one line of filler."""
        result = self.extract("stub_jsonld_posting.html")
        self.assertTrue(result.ok, result.message)
        self.assertIn("no real description", result.used)
        self.assertGreater(je.word_count(result.fields["description"]), 40)
        self.assertIn("AutoCAD", result.fields["description"])


class TestHtmlHeuristics(FixtureCase):
    def test_html_only_posting_is_split_into_sections(self):
        result = self.extract("html_only_posting.html")
        self.assertTrue(result.ok, result.message)
        self.assertIn("no schema.org JobPosting", result.used)
        self.assertGreaterEqual(je.word_count(result.fields["description"]), 40)
        self.assertGreaterEqual(je.word_count(result.fields["requirements"]), 10)
        self.assertGreaterEqual(je.word_count(result.fields["company_profile"]), 10)
        self.assertEqual(result.fields["required_experience"], "Mid-Senior level")

    def test_related_jobs_block_is_not_part_of_the_posting(self):
        result = self.extract("html_only_posting.html")
        self.assertNotIn("Data Engineer", result.fields["requirements"])


class TestRejections(FixtureCase):
    def test_listing_page_is_rejected(self):
        result = self.extract("listing_page.html")
        self.assertFalse(result.ok)
        self.assertIn("list of jobs", result.message)

    def test_captcha_page_is_rejected(self):
        result = self.extract("captcha_page.html")
        self.assertFalse(result.ok)
        self.assertIn("CAPTCHA", result.message)

    def test_javascript_shell_is_explained(self):
        """The hirist.tech case: navigation text only, the posting needs JavaScript."""
        result = self.extract("js_shell_posting.html")
        self.assertFalse(result.ok)
        self.assertIn("JavaScript", result.message)


class TestSections(unittest.TestCase):
    def test_runaway_section_is_trimmed(self):
        text = "Job description text here.\nRequirements\n" + ("word " * 900) + "\nAbout the company\nWe build things."
        description, requirements, profile = je.split_sections(text)
        self.assertLessEqual(je.word_count(requirements), je.MAX_SECTION_WORDS)
        self.assertIn("We build things", profile)

    def test_stop_heading_ends_the_posting(self):
        text = (
            "Job description\n"
            "We are hiring an engineer to build services for our platform.\n"
            "Requirements\n"
            "Five years of Python experience and strong SQL skills.\n"
            "Related jobs\n"
            "Data Engineer at Another Company in Berlin.\n"
        )
        description, requirements, profile = je.split_sections(text)
        self.assertNotIn("Another Company", requirements)
        self.assertNotIn("Another Company", description)


class TestCategoryMappers(unittest.TestCase):
    def test_employment_types(self):
        self.assertEqual(je.map_employment_type("FULL_TIME"), "Full-time")
        self.assertEqual(je.map_employment_type("CONTRACTOR"), "Contract")
        self.assertEqual(je.map_employment_type("part time"), "Part-time")

    def test_experience_and_education_rules(self):
        self.assertEqual(je.map_experience("Senior Python Engineer"), "Mid-Senior level")
        self.assertEqual(je.map_experience("5+ years of experience in Python"), "Mid-Senior level")
        self.assertEqual(je.map_experience("2-4 years of experience"), "Associate")
        self.assertEqual(je.map_experience("Freshers welcome, no experience needed"), "Entry level")
        self.assertEqual(je.map_experience("Data Science Internship"), "Internship")
        self.assertEqual(je.map_experience(""), "")
        self.assertEqual(
            je.map_from_rules("Bachelor's degree in computer science", je.EDUCATION_RULES),
            "Bachelor's Degree",
        )

    def test_normalize_category_keeps_only_trained_values(self):
        allowed = ["Information Technology and Services"]
        self.assertEqual(je.normalize_category("information technology and services", allowed), allowed[0])
        self.assertEqual(je.normalize_category("", allowed), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
